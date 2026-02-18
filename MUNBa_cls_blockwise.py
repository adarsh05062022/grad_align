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
from collections import defaultdict

TEXT_SOMETHING = "block_wise"


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
    layer_groups = defaultdict(list)
    layer_group_names = defaultdict(list)
    param_set = set(parameters)   # faster lookup

    for param_name, param in model.model.diffusion_model.named_parameters():
        if param not in param_set:
            continue

        parts = param_name.split('.')

        if "input_blocks" in param_name:
            block_id = parts[3]   # input_blocks.X
            block_name = f"input_blocks.{block_id}"

        elif "output_blocks" in param_name:
            block_id = parts[3]   # output_blocks.X
            block_name = f"output_blocks.{block_id}"

        elif "middle_block" in param_name:
            block_name = "middle_block"

        else:
            block_name = "others"

        layer_groups[block_name].append(param)
        layer_group_names[block_name].append(param_name)

    layer_groups = list(layer_groups.values())
    for block_name, names in layer_group_names.items():
        logger.info(f"\nBlock: {block_name}")
        for n in names:
            logger.info(f"   {n}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        with tqdm(total=len(forget_dl)) as time_1:
            model.train()
            remain_iter = iter(remain_dl)
            

            for i, (forget_images, forget_labels ) in enumerate(forget_dl):   
                alpha_r_vals = []
                alpha_f_vals = [] 
                grad_r_vals = []
                grad_f_vals = [] 
                model.train()
                optimizer.zero_grad()
                

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

                loss_u = criteria(forget_out, pseudo_out) 

                #####################################################
                if args.munba:
                    
                    if with_l1:
                        current_alpha = alpha * (1 - epoch / epochs)
                        l1_loss = current_alpha * l1_regularization(parameters)
                        loss_r_total = loss_r + l1_loss
                    else:
                        loss_r_total = loss_r

                    grads_r = torch.autograd.grad(loss_r_total, parameters, retain_graph=True,allow_unused=True)
                    grads_f = torch.autograd.grad(loss_u, parameters, allow_unused=True)

                    grad_r_dict = dict(zip(parameters, grads_r))
                    grad_f_dict = dict(zip(parameters, grads_f))

                    eps = 1e-8
                    xi = 1e-8

                    layerwise_updates = {}
                    alpha_r_vals = []
                    alpha_f_vals = []

                    for block in layer_groups:
                        gr_list = []
                        gf_list = []
                        valid_params = []

                        for p in block:
                            if grad_r_dict[p] is not None and grad_f_dict[p] is not None:
                                gr_list.append(grad_r_dict[p].reshape(-1))
                                gf_list.append(grad_f_dict[p].reshape(-1))
                                valid_params.append(p)

                        if len(gr_list) == 0:
                            continue

                        # ---- concatenate block grads ----
                        gr_block = torch.cat(gr_list)
                        gf_block = torch.cat(gf_list)

                        norm_gr = torch.norm(gr_block)
                        norm_gf = torch.norm(gf_block)

                        norm_gr = torch.clamp(norm_gr, min=1e-8)
                        norm_gf = torch.clamp(norm_gf, min=1e-8)

                        dot = torch.dot(gr_block, gf_block)
                        cos_phi = dot / (norm_gr * norm_gf)
                        cos_phi = torch.clamp(cos_phi, -0.999, 0.999)

                        # ---- STABLE SIN TERM ----
                        sin_sq = torch.clamp(1.0 - cos_phi ** 2, min=1e-4)

                        coeff = torch.sqrt((1.0 - cos_phi) / sin_sq)

                        # ---- Nash coefficients ----
                        alpha_r_block = coeff / norm_gr
                        alpha_f_block = beta * coeff / norm_gf   # β damping

                        alpha_r_vals.append(alpha_r_block.item())
                        alpha_f_vals.append(alpha_f_block.item())

                        grad_r_vals.append( norm_gr.item())
                        grad_f_vals.append(norm_gf.item())

                        # ---- apply same α to block ----
                        for p in valid_params:
                            g_tilde = (
                                alpha_r_block * grad_r_dict[p]
                                + alpha_f_block * grad_f_dict[p]
                            )
                            layerwise_updates[p] = g_tilde

                    avg_alpha_r = np.mean(alpha_r_vals) if alpha_r_vals else 0
                    avg_alpha_f = np.mean(alpha_f_vals) if alpha_f_vals else 0

                    avg_grad_r = np.mean(grad_r_vals)
                    avg_grad_f = np.mean(grad_f_vals)

                    loss = loss_r + loss_u
                #####################################################
                else:
                    loss = loss_r + args.lam * loss_u

                if args.munba:
                    optimizer.zero_grad()
                    for p in parameters:
                        if p in layerwise_updates:
                            p.grad = layerwise_updates[p]
                else:
                    optimizer.zero_grad()
                    loss.backward()

                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                losses.append(loss.item() / batch_size)

                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if (step+1) % 10 == 0:
                    if args.munba:
                        logger.info(f"step: {step}, avg_alpha_r: {avg_alpha_r:.4f}, avg_alpha_f: {avg_alpha_f:.4f}, avg_grad_r: {avg_grad_r:.4f}, avg_grad_f: {avg_grad_f:.4f}, loss: {loss.item():.4f}, loss_r: {loss_r.item():.4f}, loss_u: {loss_u.item():.4f}")
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
            if epoch != epochs - 1:  # save intermediate compvis checkpoints for all but last epoch
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
        default="5",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=False,default="xattn"
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
        default=5e-6,
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
    parser.add_argument("--beta", type=float, default=0.3)
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




