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



from logger.logger import setup_logger

from .mask_nsfw import compute_dual_importance_mask


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


TEXT_SOMETHING = "beta_10"

word_wear = "a photo of a person wearing clothes"
def flatten_grads(parameters, grads):

    parts = []
    for p, g in zip(parameters, grads):
        if g is not None:
            parts.append(g.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts)


def unpack_update_to_grads(parameters, flat_update, numel_list):

    offset = 0
    for p, numel in zip(parameters, numel_list):
        chunk = flat_update[offset:offset + numel].view_as(p)

        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)

        offset += numel
def _recompute_interval(epoch, epochs, steps_per_epoch):

    if epoch == 0:
        fraction = 0.25
    elif epoch <= epochs // 2:
        fraction = 0.50
    else:
        fraction = 999999

    interval = max(10, int(steps_per_epoch * fraction))
    return interval

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
    # MODEL TRAINING SETUP
    total_start_time = time.time()
    logger.info("Training started")
    # print(config_path)
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size, forget_path='./dataFolder/NSFW/SD',
        remain_path='./dataFolder/NotNSFW', image_size=image_size)
    num_forget = len(forget_dl.dataset)
    fisher_dl = forget_dl
    logger.info(f"Number of unlearning datapoints: {num_forget}")

    


    # choose parameters to train based on train_method
    parameters = []
    param_names = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
                param_names.append(name)
        # train all layers
        if train_method == "full":
            # logger.info(name)
            parameters.append(param)
            param_names.append(name)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    parameters.append(param)

    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-MUNBa-mask-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}"
    else:
        name = f"compvis-nsfw-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"

    # NSFW Removal
    word_wear = "a photo of a person wearing clothes"
    word_print = 'nsfw'.replace(" ", "")

    # TRAINING CODE
    step = 0
    
    epoch_times = []
    numel_list = [p.numel() for p in parameters]
    steps_per_epoch = len(forget_dl)
    mask = None
    mask_flat = None
    for epoch in range(epochs):
        recompute_every = _recompute_interval(epoch, epochs, steps_per_epoch)
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        with tqdm(total=len(forget_dl)) as time_1:
            
            
            remain_iter = iter(remain_dl)
            for i, forget_batch in enumerate(forget_dl):
                if step >= 0 and step % recompute_every == 0:

                    logger.info(f"Recomputing mask (step {step})")

                    model.eval()

                    mask, mask_flat = compute_dual_importance_mask(
                        model=model,
                        forget_dl=fisher_dl,
                        remain_dl=remain_dl,
                        parameters=parameters,
                        param_names=param_names,
                        descriptions=None,
                        class_to_forget=None,
                        beta=beta,
                        device=device,
                        target_density=args.mask_density,
                        lambda_tradeoff=1.0,
                        importance_variant=args.importance_variant,
                        previous_mask_flat=None if step == 0 else mask_flat,
                        ema_alpha=0.3,
                        logger=logger
                    )

                    model.train()
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
                # loss_u = -model.shared_step(forget_batch)[0]

                #####################################################
                if args.munba:
                    alpha_r = torch.tensor(0.0, device=device)
                    alpha_f = torch.tensor(0.0, device=device)
                    grad_r_norm = 0.0
                    grad_f_norm = 0.0
                    cos_value = 0.0
                    update_norm = 0.0
                    
                    
                    # compute gradient for each task
                    grads_r = torch.autograd.grad(loss_r, parameters, allow_unused=True)
                    grads_f = torch.autograd.grad(loss_u, parameters, allow_unused=True)

                    grads_r = [g.detach() if g is not None else None for g in grads_r]
                    grads_f = [g.detach() if g is not None else None for g in grads_f]

                    gr_flat = flatten_grads(parameters, grads_r)
                    gf_flat = flatten_grads(parameters, grads_f)

                    gr_masked = gr_flat[mask_flat]
                    gf_masked = gf_flat[mask_flat]

                    if gr_masked.numel() == 0:
                        del gr_flat, gf_flat, gr_masked, gf_masked
                        continue

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
                    update_full = torch.zeros_like(gr_flat)

                    update_full[mask_flat] = (
                        alpha_r * gr_masked + alpha_f * gf_masked
                    )   
                    del gr_flat, gf_flat, gr_masked, gf_masked
                    unpack_update_to_grads(parameters, update_full, numel_list)
                    del update_full
                    loss = loss_r + loss_u
                    
                #####################################################
                else:
                    loss = loss_r + args.lam * loss_u

               
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
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
                        
                    )
                    # save_history(losses, name, classes)
                if step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)
            torch.cuda.empty_cache()
            gc.collect()
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            logger.info(
                    f"Epoch {epoch+1} finished | "
                    f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)"
            )    
            model.eval()
            if epoch ==0 or (epoch + 1)%10 == 0 and epoch != epochs-1:
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
        "--train_method", help="method of training", type=str, required=False, default="full"
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=100
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

    parser.add_argument(
    "--mask_density",
    type=float,
    default=0.10
)

    parser.add_argument(
        "--importance_variant",
        type=str,
        default="both",
        choices=["ratio", "difference", "both"],
    )
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1e-4)
    ##################################### MUNBa setting #################################################
    parser.add_argument("--munba", default=True, action='store_true',)
    parser.add_argument("--beta", type=float, default=10.0)

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



