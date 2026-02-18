import os
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FORGET_DIR = "data/forget_vangogh"
PRESERVE_DIR = "data/preserve_neutral"

NUM_FORGET = 100
NUM_PRESERVE = 200

SEED = 42
GUIDANCE_SCALE = 7.5
NUM_STEPS = 50
IMG_SIZE = 512

os.makedirs(FORGET_DIR, exist_ok=True)
os.makedirs(PRESERVE_DIR, exist_ok=True)

# =========================
# PROMPTS
# =========================

FORGET_PROMPTS = [
    "a van gogh style painting of a countryside landscape",
    "a van gogh style painting of a small village at night",
    "a van gogh style painting of sunflowers in a field",
    "a van gogh style painting of a wheat field under dramatic sky",
    "a van gogh style painting of a cafe at night",
    "post-impressionist oil painting with thick impasto brush strokes",
    "oil painting in van gogh style with swirling textures",
    "expressive post-impressionist painting with bold brush strokes",
    "impasto oil painting with vibrant colors and heavy texture",
    "hand-painted post-impressionist artwork with visible brush strokes",
    "van gogh style painting of a portrait of a man",
    "van gogh style painting of a woman sitting indoors",
    "van gogh style painting of a chair in a simple room",
    "van gogh style painting of a pair of old shoes"
]

PRESERVE_PROMPTS = [
    "a photo of a countryside landscape",
    "a photo of a small village at night",
    "a photo of sunflowers in a field",
    "a photo of a wheat field under a cloudy sky",
    "a photo of a cafe at night",
    "a photo of a chair in a room",
    "a photo of a pair of shoes on the floor",
    "a photo of a city street in the evening",
    "a photo of a bedroom interior",
    "a realistic photo portrait of a man",
    "a realistic photo portrait of a woman",
    "a photo of a person sitting indoors",
    "a candid photograph of a person standing outdoors",
    "a realistic image of a landscape",
    "a realistic image of a city skyline"
]

# =========================
# LOAD MODEL
# =========================
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# =========================
# IMAGE GENERATION FUNCTION
# =========================
def generate_images(prompts, num_images, out_dir, prefix):
    count = 0
    idx = 0

    with torch.no_grad():
        for _ in tqdm(range(num_images), desc=f"Generating {prefix}"):
            prompt = prompts[idx % len(prompts)]
            idx += 1

            image = pipe(
                prompt,
                height=IMG_SIZE,
                width=IMG_SIZE,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            ).images[0]

            filename = f"{prefix}_{count:04d}.png"
            image.save(os.path.join(out_dir, filename))
            count += 1

# =========================
# RUN
# =========================
# generate_images(FORGET_PROMPTS, NUM_FORGET, FORGET_DIR, "forget")
generate_images(PRESERVE_PROMPTS, NUM_PRESERVE, PRESERVE_DIR, "preserve")

print("✅ Image generation complete.")
