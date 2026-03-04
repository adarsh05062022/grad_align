import sys
import os


import argparse
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


TEXT_SOMETHING = "setting_zero_1_percent"

def erase_parameters_by_forget_gradient(
    model,
    forget_dl,
    parameters,
    descriptions,
    class_to_forget,
    beta,
    device,
    percent=0.05,
):
    """
    Compute forget gradients and zero top-k% important weights.
    """

    print(f"\n[Gradient Erase] Computing forget gradients...")
    model.eval()
    criteria = torch.nn.MSELoss()

    accum_grads = [torch.zeros_like(p) for p in parameters]

    for forget_images, forget_labels in tqdm(forget_dl):

        forget_images = forget_images.to(device)

        forget_prompts = [
            descriptions[label] for label in forget_labels
        ]

        pseudo_prompts = [
            descriptions[(int(class_to_forget) + 1) % 10]
            for _ in forget_labels
        ]

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

        t = torch.randint(
            0,
            model.num_timesteps,
            (forget_input.shape[0],),
            device=device,
        ).long()

        noise = torch.randn_like(forget_input)

        forget_noisy = model.q_sample(
            x_start=forget_input, t=t, noise=noise
        )
        forget_out = model.apply_model(
            forget_noisy, t, forget_emb
        )

        pseudo_noisy = model.q_sample(
            x_start=pseudo_input, t=t, noise=noise
        )
        pseudo_out = model.apply_model(
            pseudo_noisy, t, pseudo_emb
        ).detach()

        loss_u = criteria(forget_out, pseudo_out) * beta

        grads_f = torch.autograd.grad(
            loss_u, parameters, allow_unused=True
        )

        for i, g in enumerate(grads_f):
            if g is not None:
                accum_grads[i] += g.detach()

    print("[Gradient Erase] Zeroing important weights...")

    with torch.no_grad():
        for p, g in zip(parameters, accum_grads):

            if g is None:
                continue

            importance = torch.abs(g)
            flat = importance.view(-1)

            k = int(percent * flat.numel())
            if k == 0:
                continue

            threshold = torch.topk(flat, k).values[-1]
            mask = importance >= threshold

            p[mask] = 0.0

    print("[Gradient Erase] Done.\n")

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

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)
    num_forget = len(forget_dl.dataset)
    logger.info(f"Number of unlearning datapoints: {num_forget}")

        

    # choose parameters to train based on train_method
    parameters = []
    for param_name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
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

    # ----------------------------------------------------
    # ONE-SHOT GRADIENT ERASE (BEFORE TRAINING)
    # ----------------------------------------------------
    erase_parameters_by_forget_gradient(
        model,
        forget_dl,
        parameters,
        descriptions,
        class_to_forget,
        beta,
        device,
        percent=0.01,   # change this %
    )

    # ----------------------------------------------------
    # Continue with normal training
    # ----------------------------------------------------
    # set model to train

    name = f"compvis-cls_{class_to_forget}-MUNBa-method_{train_method}-lr_{lr}_E{epochs}_U{num_forget}_{TEXT_SOMETHING}"
    
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
        default=4,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=3
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
        default="/storage/s25017/MUNBa/SD/models/ldm/sd-v1-4-full-ema.ckpt",
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
        default="/storage/s25017/MUNBa/SD/configs/stable-diffusion/v1-inference_nash.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="/storage/s25017/MUNBa/SD/diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="7",
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




