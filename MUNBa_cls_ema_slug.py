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


TEXT_SOMETHING = "with_ema_slug"


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)






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
    
    def identify_slug_layer(model, loss_r, loss_u):
        """
        Implements SLUG layer identification exactly as in arXiv:2407.11867
        """
        model.zero_grad()

        # ---- Forget gradients ----
        loss_u.backward(retain_graph=True)
        forget_grads = {
            n: p.grad.detach().clone()
            for n, p in model.model.diffusion_model.named_parameters()
            if p.grad is not None
        }

        model.zero_grad()

        # ---- Retain gradients ----
        loss_r.backward(retain_graph=True)
        retain_grads = {
            n: p.grad.detach().clone()
            for n, p in model.model.diffusion_model.named_parameters()
            if p.grad is not None
        }

        layer_stats = {}

        for name, param in model.model.diffusion_model.named_parameters():
            if name not in forget_grads or name not in retain_grads:
                continue

            # block-level grouping (important!)
            layer_id = ".".join(name.split(".")[:3])

            g_f = forget_grads[name]
            g_r = retain_grads[name]

            importance = g_f.norm() / (param.norm() + 1e-8)
            alignment = torch.cosine_similarity(
                g_f.flatten(), g_r.flatten(), dim=0
            )

            if layer_id not in layer_stats:
                layer_stats[layer_id] = {
                    "importance": 0.0,
                    "alignment": []
                }

            layer_stats[layer_id]["importance"] += importance.item()
            layer_stats[layer_id]["alignment"].append(alignment.item())

        # average alignment per layer
        for l in layer_stats:
            layer_stats[l]["alignment"] = np.mean(layer_stats[l]["alignment"])

        # ---- Pareto selection ----
        layers = list(layer_stats.keys())
        scores = [
            (l,
            layer_stats[l]["importance"],
            layer_stats[l]["alignment"])
            for l in layers
        ]

        # Pareto front: max importance, min alignment
        pareto = []
        for l, imp, align in scores:
            dominated = False
            for l2, imp2, align2 in scores:
                if imp2 >= imp and align2 <= align and (imp2 > imp or align2 < align):
                    dominated = True
                    break
            if not dominated:
                pareto.append((l, imp, align))

        # pick highest-importance layer on Pareto front
        pareto.sort(key=lambda x: -x[1])
        selected_layer = pareto[0][0]

        return selected_layer, pareto

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)
    num_forget = len(forget_dl.dataset)
    logger.info(f"Number of unlearning datapoints: {num_forget}")

    #### Convex Optimization Problem (bargaining game) Initialization ####
    if args.munba:
        n_tasks = 2 # K
        prvs_alpha = np.ones(n_tasks, dtype=np.float32) # alpha from iteration    
        prvs_alpha[0] = 1.0
        prvs_alpha[1] = 0.5
    #####################################################

    # choose parameters to train based on train_method
    parameters = []
    forget_params = []
    remain_params = []
    # for name, param in model.model.diffusion_model.named_parameters():
    #     # train all layers except x-attns and time_embed layers
    #     if train_method == "noxattn":
    #         if name.startswith("out.") or "attn2" in name or "time_embed" in name:
    #             pass
    #         else:
    #             # print(name)
    #             parameters.append(param)
    #     # train only self attention layers
    #     if train_method == "selfattn":
    #         if "attn1" in name:
    #             print(name)
    #             parameters.append(param)
    #     # train only x attention layers
    #     if train_method == "xattn":
    #         if "attn2" in name:
    #             # print(name)
    #             parameters.append(param)
    #     # train all layers
    #     if train_method == "full":
    #         # print(name)
    #         parameters.append(param)
    #     # train all layers except time embed layers
    #     if train_method == "notime":
    #         if not (name.startswith("out.") or "time_embed" in name):
    #             print(name)
    #             parameters.append(param)
    #     if train_method == "xlayer":
    #         if "attn2" in name:
    #             if "output_blocks.6." in name or "output_blocks.8." in name:
    #                 print(name)
    #                 parameters.append(param)
    #     if train_method == "selflayer":
    #         if "attn1" in name:
    #             if "input_blocks.4." in name or "input_blocks.7." in name:
    #                 print(name)
    #                 parameters.append(param)

    for name, param in model.model.diffusion_model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith(slug_layer):
            forget_params.append(param)

        if param in parameters:
            remain_params.append(param)
            
    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-cls_{class_to_forget}-MUNBa-mask-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"
    else:
        name = f"compvis-cls_{class_to_forget}-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"

    # TRAINING CODE
    step = 0
    epoch_times = []

    #############################
    # moving average initialization
    angle_ema = torch.tensor(0.0, device=device)
    angle_step = 0
    beta_angle = 0.70
    ema_nr = torch.tensor(0.0, device=device)
    ema_nf = torch.tensor(0.0, device=device)
    ###############################
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        with tqdm(total=len(forget_dl)) as time_1:
            model.train()
            forget_iter = iter(forget_dl)
            remain_iter = iter(remain_dl)

            for i, (images, labels) in enumerate(forget_dl):
                model.train()
                optimizer.zero_grad()
                try:
                    forget_images, forget_labels = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_dl)
                    forget_images, forget_labels = next(forget_iter)

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
                
                slug_layer, pareto_layers = identify_slug_layer(model, loss_r, loss_u)
                logger.info(f"SLUG layer selected: {slug_layer}")
                logger.info(f"SLUG Pareto layers: {pareto_layers}")

                #####################################################
                if args.munba:
                    # compute gradient for each task
                    grads = {}
                    for task, loss in zip([0, 1], [loss_r, loss_u]):
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(loss, parameters, retain_graph=True)
                        grads[task] = torch.cat([torch.flatten(g.detach()) for g in grad])

                    

                    
                   
                    logger.info(f"BETA_VALUE: - {beta_angle}")

                    gr = grads[0]
                    gf = grads[1]

                    eps = 1e-8
                    a_min = 1e-3
                    a_max = 10.0

                    # step counter
                    angle_step += 1

                    # instantaneous gradient norms
                    norm_gr = torch.norm(gr)
                    norm_gf = torch.norm(gf)

                    # EMA accumulation of gradient magnitudes
                    ema_nr = beta_angle * ema_nr + (1.0 - beta_angle) * norm_gr
                    ema_nf = beta_angle * ema_nf + (1.0 - beta_angle) * norm_gf

                    # bias correction
                    norm_gr_hat = ema_nr / (1.0 - beta_angle ** angle_step)
                    norm_gf_hat = ema_nf / (1.0 - beta_angle ** angle_step)

                    # magnitude-based step sizes
                    alpha_0 = 1.0 / (norm_gr_hat + eps)
                    alpha_1 = 1.0 / (norm_gf_hat + eps)

                    # clamp for stability
                    prvs_alpha[0] = torch.clamp(alpha_0, min=a_min, max=a_max)
                    prvs_alpha[1] = torch.clamp(alpha_1, min=a_min, max=a_max)



                    if prvs_alpha[0] > 0 and prvs_alpha[1] > 0: # Bargaining succeeded
                        loss = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]
                    else:
                        # continue
                        loss = loss_r + loss_u * 0.5
                #####################################################
                else:
                    loss = loss_r + args.lam * loss_u

                optimizer.zero_grad()
                if with_l1:
                    current_alpha = alpha * (1 - epoch / (epochs))
                    loss = loss + current_alpha * l1_regularization(parameters)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                losses.append(loss.item() / batch_size)

                optimizer.step()
                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if (step+1) % 10 == 0:
                    if args.munba:

                        logger.info(f"step: {step},norm_gr:{norm_gr},norm_gf:{norm_gf},alpha[0]: {prvs_alpha[0]:.4f},alpha[1]: {prvs_alpha[1]:.4f},loss: {loss:.4f}, loss: {loss:.4f}, loss_r: {loss_r * prvs_alpha[0]:.4f}, loss_u: {loss_u * prvs_alpha[1]:.4f}")
                    else:
                        logger.info(f"step: {step}, loss: {loss:.4f}, loss_r: {loss_r:.4f}, loss_u: {args.lam * loss_u:.4f}")
                    save_history(losses, name, classes)

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            logger.info(
                    f"Epoch {epoch+1} finished | "
                    f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)"
            )    
            model.eval()
            if epoch ==0 or (epoch + 1)%5 == 0  :
                save_model(model, name, epoch, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
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
        save_compvis=True,
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
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device, num=num,
        )


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
        default="5",
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
        "--epochs", help="epochs used to train", type=int, required=False, default=1
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
        default="0",
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




