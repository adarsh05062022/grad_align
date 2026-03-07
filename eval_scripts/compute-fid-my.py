import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance


def load_images_from_folder(folder_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)

    return images


def compute_fid(real_path, fake_path, image_size):
    fid = FrechetInceptionDistance(feature=64)

    print("Loading real images...")
    real_images = load_images_from_folder(real_path, image_size)

    print("Loading fake images...")
    fake_images = load_images_from_folder(fake_path, image_size)

    real_images = torch.stack(real_images)
    fake_images = torch.stack(fake_images)

    # Convert to uint8 [0,255]
    real_images = (real_images * 255).to(torch.uint8)
    fake_images = (fake_images * 255).to(torch.uint8)

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    score = fid.compute()
    
    print("FID Score:", score.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between two folders")
    
    parser.add_argument("--image_size", type=int, default=256)
    FAKE_PATH = "/storage/s25017/Datasets/imagenette2/val/n02979186"
    REAL_PATH = "/storage/s25017/Datasets/imagenette2/val/n02979186"

    args = parser.parse_args()

    compute_fid(REAL_PATH, FAKE_PATH, args.image_size)