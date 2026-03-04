# Top k% masking with one time nash calculation
import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_nsfw_data,
    setup_model,
    setup_forget_data,
    setup_remain_data,
)
from tqdm import tqdm

from logger.logger import setup_logger

import gc
import time
import random

import cvxpy as cp


TEXT_SOMETHING = "layer_importance_masking_5_percent_per_epoch_mask_fisher_once_nash_softmax"


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def compute_dual_importance_mask(
    model,
    forget_dl,
    remain_dl,
    parameters,
    descriptions,
    class_to_forget,
    beta,
    device,
    target_density = 0.05,        # exact % parameters to update per layer
    lambda_tradeoff = 1.0 
):
       
    print("\n[Mask] Computing dual importance mask...")
    model.eval()
    criteria = torch.nn.MSELoss()

    accum_fisher_f = [torch.zeros_like(p) for p in parameters]
    accum_fisher_r = [torch.zeros_like(p) for p in parameters]

    remain_iter = iter(remain_dl)

    for forget_images, forget_labels in tqdm(forget_dl):

        forget_images = forget_images.to(device)

        try:
            remain_images, remain_labels = next(remain_iter)
        except StopIteration:
            remain_iter = iter(remain_dl)
            remain_images, remain_labels = next(remain_iter)

        remain_images = remain_images.to(device)

        # ---- Build prompts ----
        forget_prompts = [descriptions[label] for label in forget_labels]
        pseudo_prompts = [
            descriptions[(int(class_to_forget) + 1) % 10]
            for _ in forget_labels
        ]
        remain_prompts = [descriptions[label] for label in remain_labels]

        # ---- FORGET LOSS ----
        forget_batch = {"jpg": forget_images.permute(0,2,3,1), "txt": forget_prompts}
        pseudo_batch = {"jpg": forget_images.permute(0,2,3,1), "txt": pseudo_prompts}

        forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
        pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

        t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
        noise = torch.randn_like(forget_input)

        forget_out = model.apply_model(
            model.q_sample(forget_input, t, noise),
            t,
            forget_emb,
        )
        pseudo_out = model.apply_model(
            model.q_sample(pseudo_input, t, noise),
            t,
            pseudo_emb,
        ).detach()

        loss_f = criteria(forget_out, pseudo_out) * beta

        grads_f = torch.autograd.grad(loss_f, parameters, retain_graph=False, allow_unused=True)

        del forget_out, pseudo_out, forget_input, pseudo_input, loss_f

        for i, g in enumerate(grads_f):
            if g is not None:
                accum_fisher_f[i] += g.detach() ** 2

        # ---- RETAIN LOSS ----
        remain_batch = {"jpg": remain_images.permute(0,2,3,1), "txt": remain_prompts}
        loss_r = model.shared_step(remain_batch)[0]

        grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=False, allow_unused=True)

        for i, g in enumerate(grads_r):
            if g is not None:
                accum_fisher_r[i] += g.detach() ** 2

        del loss_r, grads_r  # Explicit cleanup

    num_batches = len(forget_dl)

    for i in range(len(accum_fisher_f)):
        accum_fisher_f[i] /= num_batches
        accum_fisher_r[i] /= num_batches

    # ---- GLOBAL TOP-K MASKING ----

    all_scores = []
    layer_shapes = []

    # First pass: compute scores and store flattened versions
    for g_f, g_r in zip(accum_fisher_f, accum_fisher_r):

        if g_f is None or g_r is None:
            all_scores.append(None)
            layer_shapes.append(None)
            continue

        # Z-score normalization
        mu_f = g_f.mean()
        std_f = g_f.std()

        mu_r = g_r.mean()
        std_r = g_r.std()

        z_f = (g_f - mu_f) / (std_f + 1e-8)
        z_r = (g_r - mu_r) / (std_r + 1e-8)

        score = z_f - lambda_tradeoff * z_r

        all_scores.append(score.view(-1))
        layer_shapes.append(score.shape)

    # Concatenate all layer scores
    valid_scores = [s for s in all_scores if s is not None]
    global_scores = torch.cat(valid_scores)
    logger.info(f"Global scores computed. Total parameters considered: {global_scores.numel()}")

    

    # Global top-k
    k = int(target_density * global_scores.numel())

    if k == 0:
        return [None] * len(accum_fisher_f)

    # ---- Normalize scores ----
    scores_norm = global_scores - global_scores.mean()
    scores_norm = scores_norm / (global_scores.std() + 1e-8)

    temperature = 0.1

    # ---- Add Gumbel noise (NO multinomial) ----
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores_norm) + 1e-8) + 1e-8)

    noisy_scores = scores_norm / temperature + gumbel_noise

    # ---- Select top-k deterministically ----
    indices = torch.topk(noisy_scores, k).indices

    mask_flat = torch.zeros_like(global_scores, dtype=torch.bool)
    mask_flat[indices] = True

    # ---- Build masks per layer ----
    masks = []
    start = 0 
    for score_flat, shape in zip(all_scores, layer_shapes):

        if score_flat is None:
            masks.append(None)
            continue

        numel = score_flat.numel()
        layer_mask_flat = mask_flat[start:start + numel]
        masks.append(layer_mask_flat.view(shape))

        start += numel
    print("[Mask] Dual mask computed.\n")
    del accum_fisher_f, accum_fisher_r, all_scores, global_scores, mask_flat
    torch.cuda.empty_cache()
    return masks

def MUNBa(
    class_to_forget,
    train_method,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size,
    ddim_steps,
    with_l1,
    beta,
    alpha,
    logger
):
    
    logger.info(TEXT_SOMETHING)
    

    # MODEL TRAINING SETUP
    # print(config_path)
    total_start_time = time.time()
    logger.info("Training started")
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)
    num_forget = len(forget_dl.dataset)
    logger.info(f"Number of unlearning datapoints: {num_forget}")

        

    # choose parameters to train based on train_method
    parameters = []
    for param_name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if param_name.startswith("out.") or "attn2" in param_name or "time_embed" in param_name:
                pass
            else:
                # print(param_name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in param_name:
                print(param_name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in param_name:
                # print(param_name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(param_name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (param_name.startswith("out.") or "time_embed" in param_name):
                print(param_name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in param_name:
                if "output_blocks.6." in param_name or "output_blocks.8." in param_name:
                    print(param_name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in param_name:
                if "input_blocks.4." in param_name or "input_blocks.7." in param_name:
                    print(param_name)
                    parameters.append(param)

    # set model to train

    # ----------------------------------------------------
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    
    name = f"compvis-cls_{class_to_forget}-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"

    # TRAINING CODE
    step = 0
    epoch_times = []


    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        logger.info(f"\n===== Recomputing mask at epoch {epoch+1} =====")
        mask_start_time = time.time()
        model.eval()
        mask = compute_dual_importance_mask(
            model,
            forget_dl,
            remain_dl,
            parameters,
            descriptions,
            class_to_forget,
            beta,
            device,
            target_density = 0.05,        # exact % parameters to update per layer
            lambda_tradeoff = 1.0
        )
        mask_time = time.time() - mask_start_time
        logger.info(f"Mask recomputation finished in {mask_time:.2f}s ({mask_time/60:.2f} min)")
        logger.info(f"Dual importance masks computed for all layers. Starting training with MUNBa...")
        

        # Optional: print mask density
        total = 0
        active = 0
        for m in mask:
            if m is not None:
                total += m.numel()
                active += m.sum().item()

        density = active / total if total > 0 else 0
        logger.info(f"Mask density: {density:.6f}")
        # torch.cuda.empty_cache()
        # gc.collect()
        model.train()

        with tqdm(total=len(forget_dl)) as time_1:
            model.train()
            # forget_iter = iter(forget_dl)
            remain_iter = iter(remain_dl)
            

            for i, (forget_images, forget_labels) in enumerate(forget_dl):   
                
                model.train()
                

                try:
                    remain_images, remain_labels = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_labels = next(remain_iter)

                

                remain_prompts = [descriptions[label] for label in remain_labels]
                forget_prompts = [descriptions[label] for label in forget_labels]
                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                # 
                loss_r = model.shared_step(remain_batch)[0]

                # forget stage
                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }
                pseudo_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": pseudo_prompts,
                }
                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=model.device,).long()
                noise = torch.randn_like(forget_input, device=model.device)
                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                loss_u = criteria(forget_out, pseudo_out) * beta

                #####################################################
                if args.munba:
                    alpha_r = torch.tensor(0.0, device=device)
                    alpha_f = torch.tensor(0.0, device=device)
                    grad_r_norm = 0.0
                    grad_f_norm = 0.0
                    cos_value = 0.0
                    update_norm = 0.0
                    optimizer.zero_grad()
                    
                    if with_l1:
                        current_alpha = alpha * (1 - epoch / epochs)
                        l1_loss = current_alpha * l1_regularization(parameters)
                        loss_r_total = loss_r + l1_loss
                    else:
                        loss_r_total = loss_r

                    grads_r = torch.autograd.grad(loss_r_total, parameters,allow_unused=True)
                    grads_f = torch.autograd.grad(loss_u, parameters, allow_unused=True)

                    grads_r = [g.detach() if g is not None else None for g in grads_r]
                    grads_f = [g.detach() if g is not None else None for g in grads_f]



                    eps = 1e-8
                    xi = 1e-8


                    

                    # Collect masked gradients
                    masked_gr_list = []
                    masked_gf_list = []

                    for layer_index, (gr, gf) in enumerate(zip(grads_r, grads_f)):

                        if gr is None or gf is None:
                            continue

                        layer_mask = mask[layer_index]
                        if layer_mask is None:
                            continue
                    
                    

                        masked_gr_list.append(gr[layer_mask])
                        masked_gf_list.append(gf[layer_mask])

                    # If no masked parameters exist
                    if len(masked_gr_list) == 0:
                        continue

                    # Concatenate all masked coordinates
                    gr_masked = torch.cat(masked_gr_list)
                    gf_masked = torch.cat(masked_gf_list)

                    del masked_gr_list, masked_gf_list  # Break reference cycles

                    # Compute norms
                    norm_gr = torch.clamp(torch.norm(gr_masked), min=1e-6)
                    norm_gf = torch.clamp(torch.norm(gf_masked), min=1e-6)

                    

                    # Compute cosine
                    cos_phi = torch.dot(gr_masked, gf_masked) / (norm_gr * norm_gf)
                    cos_phi = torch.clamp(cos_phi, -1.0 + 1e-6, 1.0 - 1e-6)

                    # Gradient statistics for logging
                    grad_r_norm = norm_gr.item()
                    grad_f_norm = norm_gf.item()
                    cos_value = cos_phi.item()



                    sin_sq_phi = torch.clamp(1.0 - cos_phi**2, min=0.0)

                    if sin_sq_phi < 1e-6:
                        alpha_r = 0.5 / norm_gr
                        alpha_f = 0.5 / norm_gf
                    else:
                        alpha_r = (1.0 / norm_gr) * torch.sqrt(
                            (1.0 - cos_phi) / (sin_sq_phi + 1e-8)
                        )
                        alpha_f = (1.0 / norm_gf) * torch.sqrt(
                            sin_sq_phi * (1.0 - cos_phi)
                        )

                    update_masked = alpha_r * gr_masked + alpha_f * gf_masked
                    update_norm = torch.norm(update_masked).item()

                    # Now build updates per layer
                    for layer_index, (p, gr, gf) in enumerate(zip(parameters, grads_r, grads_f)):

                        if gr is None or gf is None:
                            continue

                        layer_mask = mask[layer_index]
                        if layer_mask is None:
                            continue

                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        p.grad[layer_mask] = (
                            alpha_r * gr[layer_mask] + alpha_f * gf[layer_mask]
                        )


                    

                    loss = loss_r + loss_u 
                #####################################################
                else:
                    loss = loss_r + args.lam * loss_u

                
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                losses.append(loss.item() / batch_size)

                step += 1
                

                if (step+1) % 20 == 0:
                    logger.info(
                        f"step: {step}, "
                        f"alpha_r: {alpha_r.item():.4f}, "
                        f"alpha_f: {alpha_f.item():.4f}, "
                        f"||g_r||: {grad_r_norm:.4f}, "
                        f"||g_f||: {grad_f_norm:.4f}, "
                        f"cos: {cos_value:.4f}, "
                        f"||update||: {update_norm:.4f}, "
                        f"loss: {loss.item():.4f}, "
                        f"loss_r: {loss_r.item():.4f}, "
                        f"loss_u: {loss_u.item():.4f}"
                    )
                    # save_history(losses, name, classes)
                torch.cuda.empty_cache()
                gc.collect()

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)
            del mask, grads_r, grads_f
            torch.cuda.empty_cache()
            gc.collect()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            logger.info(
                    f"Epoch {epoch+1} finished | "
                    f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)"
            )    
            model.eval()
            if epoch%1==0 and epoch != epochs - 1:  # save intermediate compvis checkpoints for all but last epoch
                save_model(model, name, epoch, save_compvis=False, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    total_time = time.time() - total_start_time
    logger.info("======== TRAINING FINISHED ========")
    logger.info(
        f"Total training time: "
        f"{total_time:.2f}s ({total_time/60:.2f} min | {total_time/3600:.2f} hrs)"
    )
    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=False,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=False,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    # Always save compvis temporarily if diffusers is needed
    if save_diffusers:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device, num=num,
        )
    # delete compvis checkpoint if user doesn't want it
    if (not save_compvis) and os.path.exists(path):
        os.remove(path)
        print(f"Deleted compvis checkpoint: {path}")


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)



def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="2",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=False,default="full"
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference_nash.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=256,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )

    
    ##################################### Nash setting #################################################
    parser.add_argument("--munba", default=True, action='store_true',)
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-4)

    args = parser.parse_args()
    logger, log_file = setup_logger(name=f"MUNBa_class_to_forgot{args.class_to_forget}")


    logger.info("======== MUNBa TRAINING STARTED ========")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {vars(args)}")

    setup_seed(42)

    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    MUNBa(
        classes,
        train_method,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
        args.with_l1,
        args.beta,
        args.alpha,
        logger
    )




