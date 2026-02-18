import sys
sys.path.insert(0, "/storage/s25017/MUNBa/SD/src/taming-transformers")

import os
import torch
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# ==============================
# CONFIG
# ==============================
CKPT_PATH = "models/ldm/sd-v1-4-full-ema.ckpt"
CKPT_PATH = "models/compvis-nsfw-MUNBa-method_xattn-lr_1e-05_E5_U2260/compvis-nsfw-MUNBa-method_xattn-lr_1e-05_E5_U2260.pt"
CONFIG_PATH = "configs/stable-diffusion/v1-inference_nash.yaml"

NSFW_CSV = "prompts/unsafe-prompts4703.csv"
COCO_CSV = "prompts/coco_30k.csv"

OUT_NSFW = "dataFolder/xattn_NSFW"
OUT_SAFE = "dataFolder/xattn_NotNSFW"

DEVICE = "cuda"
IMG_SIZE = 512
STEPS = 50
GUIDANCE = 7.5
MAX_IMAGES = 50   # 🔴 reduce for first run


# ==============================
# UTILS
# ==============================
def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_image(model, sampler, prompt):
    batch_size = 1
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])

    shape = [4, IMG_SIZE // 8, IMG_SIZE // 8]

    samples, _ = sampler.sample(
        S=STEPS,
        conditioning=c,
        batch_size=batch_size,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=GUIDANCE,
        unconditional_conditioning=uc,
        eta=0.0,
    )

    x_samples = model.decode_first_stage(samples)   
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

    img = (x_samples[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(img)


# ==============================
# MAIN
# ==============================
def main():
    os.makedirs(OUT_NSFW, exist_ok=True)
    os.makedirs(OUT_SAFE, exist_ok=True)

    print("Loading Stable Diffusion model...")
    model = load_model(CONFIG_PATH, CKPT_PATH, DEVICE)
    sampler = DDIMSampler(model)

    # # -------- NSFW DATA --------
    nsfw_df = pd.read_csv(NSFW_CSV)
    print(f"Generating NSFW images ({MAX_IMAGES})...")

    for i, row in tqdm(nsfw_df.iterrows(), total=MAX_IMAGES):
        if i >= MAX_IMAGES:
            break

        prompt = row["prompt"]
        class_idx = int(row.get("classidx", 0))  # 0=nude,1=naked,2=erotic,3=sexual

        img = generate_image(model, sampler, prompt)
        img.save(f"{OUT_NSFW}/{class_idx}_{i:05d}.png")

    # -------- SAFE DATA --------
    # coco_df = pd.read_csv(COCO_CSV)
    # print(f"Generating SAFE images ({MAX_IMAGES})...")

    # for i, row in tqdm(coco_df.iterrows(), total=MAX_IMAGES):
    #     if i >= MAX_IMAGES:
    #         break

    #     prompt = row["prompt"]
    #     img = generate_image(model, sampler, prompt)
    #     img.save(f"{OUT_SAFE}/img_{i:05d}.png")


if __name__ == "__main__":
    main()
