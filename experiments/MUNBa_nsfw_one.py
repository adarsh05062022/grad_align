# MUNBa_nsfw.py
    
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
)
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm

import gc
from timm.utils import AverageMeter
from timm.models.layers import trunc_normal_

import copy
import timm
import math
import time
import random


import cvxpy as cp
from logger.logger import setup_logger
TEXT_SOMETHING = "layer_importance_masking_20_precent_forget_50_percent_retain_mask"



def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)



def compute_dual_importance_mask_nsfw(
    model,
    forget_dl,
    remain_dl,
    parameters,
    beta,
    device,
    percent_forget=0.2,
    percent_retain=0.5,
):

    print("\n[Mask] Computing NSFW dual importance mask...")
    model.eval()
    criteria = torch.nn.MSELoss()

    accum_grads_f = [torch.zeros_like(p) for p in parameters]
    accum_grads_r = [torch.zeros_like(p) for p in parameters]

    remain_iter = iter(remain_dl)

    for forget_batch in tqdm(forget_dl):

        try:
            remain_batch = next(remain_iter)
        except StopIteration:
            remain_iter = iter(remain_dl)
            remain_batch = next(remain_iter)

        # -------- RETAIN --------
        loss_r = model.shared_step(remain_batch)[0]
        grads_r = torch.autograd.grad(
            loss_r, parameters, retain_graph=True, allow_unused=True
        )

        for i, g in enumerate(grads_r):
            if g is not None:
                accum_grads_r[i] += g.detach()

        # -------- FORGET --------
        forget_input, forget_emb = model.get_input(
            forget_batch, model.first_stage_key
        )

        pseudo_prompts = ["a photo of a person wearing clothes"] * forget_batch['jpg'].size(0)

        pseudo_batch = {
            "jpg": forget_batch['jpg'],
            "txt": pseudo_prompts,
        }

        pseudo_input, pseudo_emb = model.get_input(
            pseudo_batch, model.first_stage_key
        )

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

        loss_f  = criteria(forget_out, pseudo_out) * beta

        grads_f = torch.autograd.grad(
            loss_f, parameters, allow_unused=True
        )

        for i, g in enumerate(grads_f):
            if g is not None:
                accum_grads_f[i] += g.detach()

    # -------- Build Masks --------
    masks = []

    for g_f, g_r in zip(accum_grads_f, accum_grads_r):

        if g_f is None or g_r is None:
            masks.append(None)
            continue

        imp_f = torch.abs(g_f)
        imp_r = torch.abs(g_r)

        flat_f = imp_f.view(-1)
        flat_r = imp_r.view(-1)

        k_f = int(percent_forget * flat_f.numel())
        if k_f == 0:
            masks.append(torch.zeros_like(g_f, dtype=torch.bool))
            continue

        thresh_f = torch.topk(flat_f, k_f).values[-1]
        mask_f = imp_f >= thresh_f

        k_r = int(percent_retain * flat_r.numel())
        thresh_r = torch.topk(flat_r, k_r, largest=False).values[-1]
        mask_r = imp_r <= thresh_r

        final_mask = mask_f & mask_r
        masks.append(final_mask)

    print("[Mask] NSFW dual mask computed.\n")
    return masks



word_wear = "a photo of a person wearing clothes"


def MUNBa(classes,
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
    total_start_time = time.time()
    logger.info("Training started")
    # print(config_path)
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size, forget_path='./dataFolder/NSFW',
        remain_path='./dataFolder/NotNSFW', image_size=image_size)
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
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in param_name:
                # logger.info(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in param_name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # logger.info(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (param_name.startswith("out.") or "time_embed" in param_name):
                # logger.info(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in param_name:
                if "output_blocks.6." in param_name or "output_blocks.8." in param_name:
                    # logger.info(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in param_name:
                if "input_blocks.4." in param_name or "input_blocks.7." in param_name:
                    # logger.info(name)
                    parameters.append(param)

    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)

    name = f"compvis-nsfw-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"

    # NSFW Removal
    word_wear = "a photo of a person wearing clothes"
    word_print = 'nsfw'.replace(" ", "")

    forget_masks = compute_dual_importance_mask_nsfw(
        model,
        forget_dl,
        remain_dl,
        parameters,
        beta,
        device,
        percent_forget=0.2,
        percent_retain=0.5,
    )

    total = 0
    active = 0
    for m in forget_masks:
        if m is not None:
            total += m.numel()
            active += m.sum().item()
    density = active / total if total > 0 else 0
    logger.info(f"Mask density: {density:.6f}")

    # TRAINING CODE
    step = 0
    
    epoch_times = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        with tqdm(total=len(forget_dl)) as time_1:
            model.train()

            remain_iter = iter(remain_dl)
            for i, forget_batch in enumerate(forget_dl):
                alpha_r_vals = []
                alpha_f_vals = [] 
                grad_r_vals = []
                grad_f_vals = [] 
                optimizer.zero_grad()
                
                try:
                    remain_batch = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_batch = next(remain_iter)

                loss_r = model.shared_step(remain_batch)[0]

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_prompts = [word_wear] * forget_batch['jpg'].size(0)
                pseudo_batch = {
                    "jpg": forget_batch['jpg'],
                    "txt": pseudo_prompts,
                }
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

                # ----------------------------
                # Optional L1
                # ----------------------------
                if with_l1:
                    current_alpha = alpha * (1 - epoch / epochs)
                    loss_r_total = loss_r + current_alpha * l1_regularization(parameters)
                else:
                    loss_r_total = loss_r

                # ----------------------------
                # Compute Gradients
                # ----------------------------
                grads_r = torch.autograd.grad(
                    loss_r_total,
                    parameters,
                    retain_graph=True,
                    allow_unused=True
                )

                grads_f = torch.autograd.grad(
                    loss_u,
                    parameters,
                    allow_unused=True
                )

                # ----------------------------
                # Layerwise Nash Combination
                # ----------------------------
                layerwise_updates = []

                for layer_index, (gr, gf) in enumerate(zip(grads_r, grads_f)):

                    if gr is None or gf is None:
                        layerwise_updates.append(None)
                        continue

                    gr_flat = gr.view(-1)
                    gf_flat = gf.view(-1)

                    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
                    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

                    cos_phi = torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf)
                    cos_phi = torch.clamp(cos_phi, -1.0 + 1e-6, 1.0 - 1e-6)

                    sin_sq_phi = 1.0 - cos_phi**2

                    if sin_sq_phi < 1e-6:
                        alpha_r_l = 0.5 / norm_gr
                        alpha_f_l = 0.5 / norm_gf
                    else:
                        coeff = torch.sqrt((1.0 - cos_phi) / (sin_sq_phi + 1e-8))
                        alpha_r_l = coeff / norm_gr
                        alpha_f_l = coeff / norm_gf

                    alpha_r_vals.append(alpha_r_l.item())
                    alpha_f_vals.append(alpha_f_l.item())

                    grad_r_vals.append(alpha_r_l.item() * norm_gr.item())
                    grad_f_vals.append(alpha_f_l.item() * norm_gf.item())

                    g_tilde = alpha_r_l * gr + alpha_f_l * gf

                    mask = forget_masks[layer_index]

                    if mask is not None:
                        g_masked = torch.zeros_like(g_tilde)
                        g_masked[mask] = g_tilde[mask]
                        layerwise_updates.append(g_masked)
                    else:
                        layerwise_updates.append(None)
                    
                avg_alpha_r = np.mean(alpha_r_vals) if len(alpha_r_vals) > 0 else 0
                avg_alpha_f = np.mean(alpha_f_vals) if len(alpha_f_vals) > 0 else 0

                avg_grad_r = np.mean(grad_r_vals)
                avg_grad_f = np.mean(grad_f_vals)

                optimizer.zero_grad()

                for p, g_tilde in zip(parameters, layerwise_updates):
                    if g_tilde is not None:
                        p.grad = g_tilde
                

                nn.utils.clip_grad_norm_(parameters, 1.0)

                optimizer.step()
                loss = loss_r + loss_u
                losses.append(loss.item() / batch_size)

                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if (step+1) % 10 == 0:
                    if args.munba:
                        logger.info(f"step: {step}, avg_alpha_r: {avg_alpha_r:.4f}, avg_alpha_f: {avg_alpha_f:.4f}, avg_grad_r: {avg_grad_r:.4f}, avg_grad_f: {avg_grad_f:.4f}, loss: {loss.item():.4f}, loss_r: {loss_r.item():.4f}, loss_u: {loss_u.item():.4f}")
                    else:
                        logger.info(f"step: {step}, loss: {loss:.4f}, loss_r: {loss_r:.4f}, loss_u: {args.lam * loss_u:.4f}")
                    save_history(losses, name, word_print)

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
            if epoch%1==0 and epoch != epochs - 1 :
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
    save_history(losses, name, word_print)


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
    logger.info("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    logger, log_file = setup_logger(name="MUNBa_NSFw")
    

    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="0",
    )
    
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=False, default="xattn"
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=4,
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
        default="2",
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
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1e-4)
    ##################################### MUNBa setting #################################################
    parser.add_argument("--munba", default=True, action='store_true',)
    parser.add_argument("--beta", type=float, default=100.0)

    args = parser.parse_args()

    logger.info("======== MUNBa TRAINING STARTED ========")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {vars(args)}")

    setup_seed(42)

    classes = int(args.class_to_forget)
    logger.info(classes)
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



