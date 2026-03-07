# evaluate_models.py

import os
import json
import time
import argparse

from generate_images import generate_images
from compute_fid import compute_fid
from datetime import datetime
import gc
import torch


def evaluate_checkpoint(
    checkpoint_path,
    prompts_path,
    save_root,
    class_to_forget,
    image_size,
    num_samples,
    device,
    guidance_scale,
    ddim_steps,
):
    """
    Generate images and compute FID for one checkpoint.
    Returns dict with evaluation results.
    """

    model_name = os.path.basename(checkpoint_path)

    print(f"\n===== Evaluating: {model_name} =====")


    # 1️⃣ Generate Images
    generate_images(
        model_name=checkpoint_path,
        prompts_path=prompts_path,
        save_path=save_root,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        from_case=0,
    )

    # Folder where images were saved
    generated_folder = os.path.join(save_root, model_name)

    # 2️⃣ Compute FID
    print("Computing FID...")
    fid_value = compute_fid(
        class_to_forget=class_to_forget,
        path=generated_folder,
        image_size=image_size,
    )

    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "model_name": model_name,
        "num_samples": num_samples,
        "fid": float(fid_value),
        "last_execution": end_datetime,
    }
    del fid_value
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main(args):

    # Load checkpoint list
    with open(args.checkpoints_txt, "r") as f:
        checkpoints = [line.strip() for line in f if line.strip()]

    results = []

    for ckpt in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path=ckpt,
            prompts_path=args.prompts_path,
            save_root=args.save_path,
            class_to_forget=args.class_to_forget,
            image_size=args.image_size,
            num_samples=args.num_samples,
            device=args.device,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
        )
        # Load previous results
        if os.path.exists(args.output_json):
            with open(args.output_json, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        # Append new result
        existing.append(result)

        # Save immediately
        with open(args.output_json, "w") as f:
            json.dump(existing, f, indent=4)

        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    

    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_txt", type=str, required=False, default="/storage/s25017/MUNBa/SD/eval_scripts_cifar/CLASS/checkpoints_to_evaluate.txt")
    parser.add_argument("--prompts_path", type=str, required=False, default="/storage/s25017/MUNBa/SD/eval_scripts_cifar/CLASS/prompts.csv")
    parser.add_argument("--save_path", type=str, required=False, default="/storage/s25017/MUNBa/SD/eval_scripts_cifar/CLASS/generated_images")
    parser.add_argument("--output_json", type=str, default="CLASS/results.json")

    parser.add_argument("--class_to_forget", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ddim_steps", type=int, default=50)

    args = parser.parse_args()

    main(args)