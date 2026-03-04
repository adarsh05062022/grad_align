import os
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_MODEL = "CompVis/stable-diffusion-v1-4"

PROMPT_FILE = "prompts_nsfw.txt"
# PROMPT_FILE = "classwise_prompts.txt"
# PROMPT_FILE = "prompts_style.txt"

SEED = 43
GUIDANCE_SCALE = 7.5
STEPS = 50
DEVICE = "cuda"
DTYPE = torch.float16

# =========================
# EXPERIMENT SETUP
# =========================
RUNS = [
    # {
    #     "name": "base",
    #     "unet_ckpt": None,
    #     "out_dir": "images_base",
    # },
    {
        "name": "unlearned",
        "unet_ckpt": "/storage/s25017/MUNBa/SD/models/compvis-nsfw-MUNBa-method_xattn-lr_1e-05_E5_U2256_layer_importance_masking_40_precent_forget_60_percent_retain_mask/diffusers-nsfw-MUNBa-method_xattn-lr_1e-05_E5_U2256_layer_importance_masking_40_precent_forget_60_percent_retain_mask-epoch_2.pt",
        "out_dir": "images_unlearned",
    },
]

# =========================
# Load prompts
# =========================
with open(PROMPT_FILE) as f:
    prompts = [l.strip() for l in f if l.strip()]

print(f"Loaded {len(prompts)} prompts")

# =========================
# Load base pipeline ONCE
# =========================
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None,  # IMPORTANT for NSFW evaluation
).to(DEVICE)

pipe.set_progress_bar_config(disable=True)

# Freeze non-UNet modules (good practice)
pipe.text_encoder.eval()
pipe.vae.eval()

# =========================
# Run experiments
# =========================
for run in RUNS:
    print(f"\n========== Generating: {run['name']} ==========")

    # Load UNet weights if provided
    if run["unet_ckpt"] is not None:
        print(f"Loading UNet from: {run['unet_ckpt']}")
        unet_state = torch.load(run["unet_ckpt"], map_location="cpu")
        pipe.unet.load_state_dict(unet_state)
    else:
        print("Using base UNet (no unlearning)")

    pipe.unet.eval()

    # Output directory
    os.makedirs(run["out_dir"], exist_ok=True)

    # IMPORTANT: reset generator for each run
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    for idx, prompt in enumerate(tqdm(prompts, desc=f"{run['name']} images")):
        image = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

        image.save(os.path.join(run["out_dir"], f"{idx:04d}.png"))

    print(f"Saved {len(prompts)} images to {run['out_dir']}")
