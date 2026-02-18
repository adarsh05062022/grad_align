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


TEXT_SOMETHING = "initial_formula_with_ema"


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def _stop_criteria(gtg, alpha_t, alpha_param, prvs_alpha_param):
    return (
        (alpha_param.value is None)
        or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
        or (
            np.linalg.norm(alpha_param.value - prvs_alpha_param.value)
            < 1e-3
        )
    )


def return_weights(grads, prvs_alpha, G_param, normalization_factor_param,
                   alpha_param, prvs_alpha_param, prob):
    G = torch.stack(tuple(v for v in grads.values()))
    GTG = torch.mm(G, G.t())
    normalization_factor = (
        torch.norm(GTG).detach().cpu().numpy().reshape((1,)) + 1e-6
        )
    if (np.isnan(normalization_factor) | np.isinf(normalization_factor)).any():
        normalization_factor = np.array([1.0])
    GTG = GTG / normalization_factor.item()
    gtg = GTG.cpu().detach().numpy()
    G_param.value = gtg
    normalization_factor_param.value = normalization_factor

    optim_niter=100
    alpha_t = prvs_alpha
    for _ in range(optim_niter):
        try:
            alpha_param.value = alpha_t
            prvs_alpha_param.value = alpha_t
            # try:
            prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
        except:
            alpha_param.value = prvs_alpha_param.value

        if _stop_criteria(gtg, alpha_t, alpha_param, prvs_alpha_param):
            break

        alpha_t = alpha_param.value
    if alpha_t is not None and not (np.isnan(alpha_t) | np.isinf(alpha_t)).any():
        return alpha_t
    else:
        return prvs_alpha


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

    #### Convex Optimization Problem (bargaining game) Initialization ####
    if args.munba:
        n_tasks = 2 # K
        init_gtg = np.eye(n_tasks) # G^T G: gradient matrix product, shape: [K, K]
        G_param = cp.Parameter(shape=(n_tasks, n_tasks), value=init_gtg) # will be updated in-loop with the current GTG
        normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0])) # will be updated in-loop with torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        alpha_param = cp.Variable(shape=(n_tasks,), nonneg=True) # current alpha, shape: [K,]
        prvs_alpha = np.ones(n_tasks, dtype=np.float32) # alpha from iteration
        prvs_alpha_param = cp.Parameter(shape=(n_tasks,), value=prvs_alpha) # shape: [K,]

        # First-order approximation of Phi_alpha using Phi_alpha_(tao)
        G_prvs_alpha = G_param @ prvs_alpha_param
        prvs_phi_tag = 1 / prvs_alpha_param + (1 / G_prvs_alpha) @ G_param
        phi_alpha = prvs_phi_tag @ (alpha_param - prvs_alpha_param)

        # Beta(alpha)
        G_alpha = G_param @ alpha_param
        # Constraint: For any i, Phi_i_alpha >= 0
        constraint = []
        for i in range(n_tasks):
            constraint.append(
                -cp.log(alpha_param[i] * normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )

        # Objective: Minimize sum(Phi_alpha) + Phi_alpha / normalization_factor_param
        obj = cp.Minimize(
            cp.sum(G_alpha) + phi_alpha / normalization_factor_param
        )
        prob = cp.Problem(obj, constraint)
        logger.info("Convex optimization problem initialized.")
        # prvs_alpha[0] = 1.0
        # prvs_alpha[1] = 0.5
    #####################################################

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                print(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    print(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    print(name)
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

    #############################
    # moving average initialization
    cos_ema = None
    beta = 0.99
    eps = 1e-8
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

                #####################################################
                if args.munba:
                    # compute gradient for each task
                    grads = {}
                    for task, loss in zip([0, 1], [loss_r, loss_u]):
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(loss, parameters, retain_graph=True)
                        grads[task] = torch.cat([torch.flatten(g.detach()) for g in grad])

                    # ############# [1] Choose to use the iterative solution
                    # prvs_alpha = return_weights(grads, prvs_alpha, G_param, normalization_factor_param,
                    #                             alpha_param, prvs_alpha_param, prob)
                    # print(f'prvs_alpha: {prvs_alpha}')
                    # if np.all(prvs_alpha == 1): # Bargaining failed
                    #     # continue
                    #     loss = loss_r + loss_u * 0.5
                    # else:
                    #     loss = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]

                    ############ [2] Choose to use the closed-form solution

                    # Original

                    # g1 = torch.dot(grads[0], grads[0])
                    # g2 = torch.dot(grads[0], grads[1])
                    # g3 = torch.dot(grads[1], grads[1])
                    # prvs_alpha[0] = torch.sqrt( (g1*g3 - g2*torch.sqrt(g1*g3)) / (g1*g1*g3 - g1*g2*g2 + 1e-8) )
                    # prvs_alpha[1] = (1 - g1 * prvs_alpha[0] * prvs_alpha[0]) / (g2*prvs_alpha[0] + 1e-8)
                    
                    
                    gr = grads[0]
                    gf = grads[1]

                    # inner products
                    g1 = torch.dot(gr, gr)
                    g3 = torch.dot(gf, gf)
                    g2 = torch.dot(gr, gf)

                    # instantaneous cosine
                    cos_t = g2 / (torch.sqrt(g1 * g3) + eps)
                    cos_t = torch.clamp(cos_t, -1.0, 1.0)

                    # EMA of cosine (angle accumulation)
                    if cos_ema is None:
                        cos_ema = cos_t.detach()
                    else:
                        cos_ema = beta * cos_ema + (1.0 - beta) * cos_t.detach()

                    # reconstruct smoothed g2
                    g2_ema = cos_ema * torch.sqrt(g1 * g3)

                    # compute prvs_alpha using smoothed geometry
                    prvs_alpha[0] = torch.sqrt(
                        (g1 * g3 - g2_ema * torch.sqrt(g1 * g3)) /
                        (g1 * g1 * g3 - g1 * g2_ema * g2_ema + eps)
                    )

                    prvs_alpha[1] = (1 - g1 * prvs_alpha[0] * prvs_alpha[0]) / (g2_ema * prvs_alpha[0] + eps)


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
                        logger.info(f"step: {step},alpha[0]: {prvs_alpha[0]:.4f},alpha[1]: {prvs_alpha[1]:.4f},loss: {loss:.4f}, loss: {loss:.4f}, loss_r: {loss_r * prvs_alpha[0]:.4f}, loss_u: {loss_u * prvs_alpha[1]:.4f}")

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
            if epoch ==0 or (epoch + 1)%5 == 0 :
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
        default=1,
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
        default="6",
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




