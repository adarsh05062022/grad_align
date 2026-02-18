import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ================= CONFIG =================
BASE_DIR = "images_base"
UNLEARNED_DIR = "images_unlearned"
PROMPT_FILE = "prompts_style.txt"
PROMPT_FILE = "prompts_nsfw.txt"
PROMPT_FILE = "classwise_prompts.txt"

OUTPUT_DIR = "merged_landscape"

IMAGES_PER_FIG = 10
FONT_SIZE = 22
MAX_CHARS_PER_LINE = 35
CAPTION_HEIGHT = 120
MARGIN = 10
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prompts
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

base_imgs = sorted(os.listdir(BASE_DIR))
un_imgs = sorted(os.listdir(UNLEARNED_DIR))

assert base_imgs == un_imgs, "Base & unlearned filenames must match"
assert len(base_imgs) == len(prompts), "Prompt count mismatch"

# Load font
try:
    font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()

def draw_prompt(draw, text, x, y, w):
    lines = textwrap.wrap(text, MAX_CHARS_PER_LINE)
    for i, line in enumerate(lines):
        draw.text((x, y + i * (FONT_SIZE + 4)), line, fill="black", font=font)

# Process in groups of 5
for idx in range(0, len(base_imgs), IMAGES_PER_FIG):
    batch_imgs = base_imgs[idx:idx + IMAGES_PER_FIG]
    batch_prompts = prompts[idx:idx + IMAGES_PER_FIG]

    if len(batch_imgs) < IMAGES_PER_FIG:
        break

    sample_img = Image.open(os.path.join(BASE_DIR, batch_imgs[0]))
    w, h = sample_img.size

    canvas_w = IMAGES_PER_FIG * w
    canvas_h = CAPTION_HEIGHT + (2 * h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i in range(IMAGES_PER_FIG):
        x = i * w

        # Prompt
        draw_prompt(draw, f"Prompt: {batch_prompts[i]}", x + MARGIN, 5, w)

        # Base image
        base_img = Image.open(os.path.join(BASE_DIR, batch_imgs[i])).convert("RGB")
        canvas.paste(base_img, (x, CAPTION_HEIGHT))

        # Unlearned image
        un_img = Image.open(os.path.join(UNLEARNED_DIR, batch_imgs[i])).convert("RGB")
        canvas.paste(un_img, (x, CAPTION_HEIGHT + h))

        # Labels
        draw.text((x + 5, CAPTION_HEIGHT + 5), "Base", fill="red", font=font)
        draw.text((x + 5, CAPTION_HEIGHT + h + 5), "Unlearned", fill="green", font=font)

    out_path = os.path.join(
        OUTPUT_DIR, f"with_only_norm_accumulation_beta_0-95_E5_.png"
    )
    canvas.save(out_path)
    print(f"Saved {out_path}")

print("✅ Landscape comparison images generated.")
