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

from .mask import compute_dual_importance_mask




TEXT_SOMETHING = "topk15_pseudo_an_image"  # used in logging and naming outputs, change to reflect your experiment setting


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def flatten_grads(parameters, grads):
    """
    Flatten ALL grads into ONE contiguous tensor.
    Zero-fills missing grads so indexing stays aligned with global_mask_flat.
    """
    parts = []
    for p, g in zip(parameters, grads):
        if g is not None:
            parts.append(g.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts)  # single contiguous tensor


def unpack_update_to_grads(parameters, flat_update, numel_list):
    """
    Write flat_update back into p.grad in one pass.
    """
    offset = 0
    for p, numel in zip(parameters, numel_list):
        chunk = flat_update[offset:offset + numel].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += numel

def _recompute_interval(epoch, epochs, steps_per_epoch):
    """
    Recompute interval in STEPS, scaled by epoch size.
    Target: recompute every ~25% of epoch-0, ~50% of mid epochs.
    
    Example:
        steps_per_epoch=900  → epoch0: every 225 steps, mid: every 450 steps
        steps_per_epoch=50   → epoch0: every 12 steps,  mid: every 25 steps
    """
    if epoch == 0:
        fraction = 0.25          # recompute 4x in first epoch (most volatile)
    elif epoch <= epochs // 2:
        fraction = 0.50          # recompute 2x in mid epochs
    else:
        fraction = 999999        # once per epoch in late phase (model stabilising)

    interval = max(10, int(steps_per_epoch * fraction))
    return interval

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
    fisher_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)
    num_forget = len(forget_dl.dataset)
    logger.info(f"Number of unlearning datapoints: {num_forget}")

        

    # choose parameters to train based on train_method
    parameters = []
    param_names = []
    for param_name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if param_name.startswith("out.") or "attn2" in param_name or "time_embed" in param_name:
                pass
            else:
                # print(param_name)
                parameters.append(param)
                param_names.append(param_name)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in param_name:
                print(param_name)
                parameters.append(param)
                param_names.append(param_name)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in param_name:
                # print(param_name)
                parameters.append(param)
                param_names.append(param_name)
        # train all layers
        if train_method == "full":
            # print(param_name)
            parameters.append(param)
            param_names.append(param_name)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (param_name.startswith("out.") or "time_embed" in param_name):
                print(param_name)
                parameters.append(param)
                param_names.append(param_name)
        if train_method == "xlayer":
            if "attn2" in param_name:
                if "output_blocks.6." in param_name or "output_blocks.8." in param_name:
                    print(param_name)
                    parameters.append(param)
                    param_names.append(param_name)
        if train_method == "selflayer":
            if "attn1" in param_name:
                if "input_blocks.4." in param_name or "input_blocks.7." in param_name:
                    print(param_name)
                    parameters.append(param)
                    param_names.append(param_name)

    # set model to train

    # ----------------------------------------------------
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    
    name = f"compvis-cls_{class_to_forget}-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"

    # TRAINING CODE
    step = 0
    epoch_times = []
    # Pre-compute numel_list once — reused every step
    numel_list = [p.numel() for p in parameters]

    # Compute mask ONCE before training, not inside epoch loop
    logger.info("Computing importance mask (once, before training)...")
    # model.eval()
    # mask, mask_flat = compute_dual_importance_mask(
    #     model,
    #     forget_dl,
    #     remain_dl,
    #     parameters,
    #     descriptions,
    #     class_to_forget,
    #     beta,
    #     device,
    #     target_density=0.10,  # target 15% parameters active
    #     lambda_tradeoff=1.0
    # )
    
    

    # total = mask_flat.numel()
    # active = mask_flat.sum().item()
    # logger.info(f"Mask density: {active/total:.6f} ({active}/{total} params active)")
    # model.train()



    # logger.info(f"Dual importance masks computed for all layers. Starting training with MUNBa...")
    steps_per_epoch = len(forget_dl)   # already accounts for batch size
    logger.info(f"Steps per epoch: {steps_per_epoch} | Batch size: {batch_size}")
    max_fisher_batches = min(80, steps_per_epoch // 3)
    logger.info(f"Fisher cap: {max_fisher_batches} batches (steps_per_epoch={steps_per_epoch})")


    for epoch in range(epochs):
        recompute_every = _recompute_interval(epoch, epochs, steps_per_epoch)
        logger.info(f"Epoch {epoch+1}: mask recompute every {recompute_every} steps")
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")

        # ── EMA mask recomputation every epoch after the first ────────────────
        # The model weights have shifted so the importance landscape has changed.
        # We recompute and blend with the previous mask to avoid sudden thrashing.
        
            

        with tqdm(total=len(forget_dl)) as time_1:
            # model.train()
            # forget_iter = iter(forget_dl)
            remain_iter = iter(remain_dl)
            

            for i, (forget_images, forget_labels) in enumerate(forget_dl):   
                if step>=0 and step % recompute_every == 0:
                    logger.info(f"Recomputing mask with EMA (Step {step})...")
                    mask_start_time = time.time()

                    model.eval()
                    mask, mask_flat = compute_dual_importance_mask(
                        model=model,
                        forget_dl=fisher_dl,  # use separate dataloader for fisher computation to avoid exhausting forget_dl iterator
                        remain_dl=remain_dl,
                        parameters=parameters,
                        param_names=param_names,
                        descriptions=descriptions,
                        class_to_forget=class_to_forget,
                        beta=beta,
                        device=device,
                        target_density=0.15,
                        lambda_tradeoff=1.0,
                        importance_variant=args.importance_variant,
                        previous_mask_flat=None if step == 0 else mask_flat,  # FIX: pass previous mask for EMA
                        ema_alpha=0.3,
                        logger=logger,
                        max_fisher_batches=max_fisher_batches
                    )
                    mask_time = time.time() - mask_start_time
                    logger.info(f"Mask computed in {mask_time:.2f}s ({mask_time/60:.2f} min)")
                    model.train()
                
                

                try:
                    remain_images, remain_labels = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images, remain_labels = next(remain_iter)

                

                remain_prompts = [descriptions[label] for label in remain_labels]
                forget_prompts = [descriptions[label] for label in forget_labels]
                #   chnage this 
                # pseudo_prompts = [
                #     descriptions[(int(class_to_forget) + 1) % 10]
                #     for label in forget_labels
                # ]
                pseudo_prompts = [
                    "an abstract texture"
                    for label in forget_labels
                ]
                # PSEUDO_TARGET = "a blurry unrecognizable texture with no specific features"
                # pseudo_prompts = [PSEUDO_TARGET for _ in forget_labels]

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



                    # FIX: ONE flatten → ONE gather → compute Nash → ONE scatter
                    # Replaces 2*N gather + N cat + 2*N scatter across layers

                    # Step A: flatten all grads into two contiguous vectors
                    gr_flat = flatten_grads(parameters, grads_r)
                    gf_flat = flatten_grads(parameters, grads_f)

                    # Step B: ONE boolean index on contiguous memory
                    gr_masked = gr_flat[mask_flat]
                    gf_masked = gf_flat[mask_flat]

                    if gr_masked.numel() == 0:
                        del gr_flat, gf_flat, gr_masked, gf_masked
                        continue

                    # Step C: Nash weight computation (identical logic)
                    norm_gr = torch.clamp(torch.norm(gr_masked), min=1e-6)
                    norm_gf = torch.clamp(torch.norm(gf_masked), min=1e-6)
                    cos_phi = torch.clamp(
                        torch.dot(gr_masked, gf_masked) / (norm_gr * norm_gf),
                        -1.0 + 1e-6, 1.0 - 1e-6
                    )
                    sin_sq_phi = torch.clamp(1.0 - cos_phi ** 2, min=0.0)

                    grad_r_norm = norm_gr.item()
                    grad_f_norm = norm_gf.item()
                    cos_value = cos_phi.item()

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

                    update_norm = torch.norm(alpha_r * gr_masked + alpha_f * gf_masked).item()

                    # Step D: build full update vector, zeros outside mask
                    update_full = torch.zeros_like(gr_flat)
                    update_full[mask_flat] = alpha_r * gr_masked + alpha_f * gf_masked

                    del gr_flat, gf_flat, gr_masked, gf_masked

                    # Step E: ONE unpack into p.grad — replaces N per-layer scatter ops
                    unpack_update_to_grads(parameters, update_full, numel_list)
                    del update_full


                    

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
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)
            del grads_r, grads_f

            torch.cuda.empty_cache()
            gc.collect()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            logger.info(
                    f"Epoch {epoch+1} finished | "
                    f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)"
            )    
            model.eval()
            if (epoch+1)%3==0 and epoch != epochs - 1:  # save intermediate compvis checkpoints for all but last epoch
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
        default="4",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=False,default="full"
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=10
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
        default="3",
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

    parser.add_argument(
    "--importance_variant",
    type=str,
    default="both",
    choices=["ratio", "difference", "both"],
    )

    
    ##################################### Nash setting #################################################
    parser.add_argument("--munba", default=True, action='store_true',)
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=0.5)

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




