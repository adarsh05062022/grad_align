import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG (EDIT)
# =========================
IMAGE_DIRS = ["images_base","images_unlearned"]   # or images_unlearned
# PROMPT_FILE = "classwise_prompts.txt"
PROMPT_FILE = "prompts_nsfw.txt"
CLIP_MODEL = "ViT-B/32"
DEVICE = "cuda"

# =========================
# Load CLIP
# =========================
model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
model.eval()

# =========================
# Load prompts
# =========================
with open(PROMPT_FILE, "r") as f:
    prompts = [l.strip() for l in f if l.strip()]

text_tokens = clip.tokenize(prompts).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# =========================
# Load images & compute score
# =========================
for IMAGE_DIR in IMAGE_DIRS:
    scores = []

    image_files = sorted(os.listdir(IMAGE_DIR))

    for img_name in tqdm(image_files, desc="Computing CLIP score"):
        if not img_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = preprocess(
            Image.open(img_path).convert("RGB")
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity (image vs all prompts)
            # score = (image_features @ text_features.T).mean().item()

            # similarity to each class prompt
            sims = (image_features @ text_features.T).squeeze(0)

            # score for FORGOTTEN class only (index 0)
            score = sims[2].item()
            
            # 🔴 NSFW score for THIS image
            # score = sims.mean().item()
        scores.append((img_name, score))

            

    # =========================
    # Report
    # =========================
    mean_score = sum(s for _, s in scores) / len(scores)

    print("=" * 60)
    print(f"Images directory : {IMAGE_DIR}")
    print(f"Num images       : {len(scores)}")
    print(f"CLIP model       : {CLIP_MODEL}")
    print(f"Scores       : {scores}")
    print(f"Mean CLIP score  : {mean_score:.4f}")
    print("=" * 60)
