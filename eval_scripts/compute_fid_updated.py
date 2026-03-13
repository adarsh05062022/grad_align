"""
compute_fid.py — FID computation for single or multi-class forgetting.

IMPORTANT: Tensors from dataset.py are in [-1, 1] (Normalize([0.5],[0.5])).
They MUST be denormalised back to [0, 255] uint8 before FID update.
Skipping this step produces completely wrong FID numbers.
"""

import argparse
from typing import List, Union

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset import setup_fid_data, setup_fid_data_i2p

ClassSpec = Union[int, List[int]]


def _denorm_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a [-1, 1] float tensor to [0, 255] uint8.
    Input shape: (N, C, H, W) or (C, H, W).
    """
    tensor = (tensor * 0.5 + 0.5).clamp(0.0, 1.0)  # → [0, 1]
    tensor = (tensor * 255).to(torch.uint8)
    return tensor


def compute_fid(class_to_forget: ClassSpec,
                path: str,
                image_size: int,
                feature: int = 64) -> float:
    """
    Compute FID between real Imagenette images (retain classes only)
    and generated images in `path` (retain classes only).

    Args:
        class_to_forget : int or list[int]  — classes excluded from both sets.
        path            : directory of generated .png files.
        image_size      : spatial resolution for resizing.
        feature         : InceptionV3 feature layer dimension (64 / 192 / 768 / 2048).

    Returns:
        FID value as float.
    """
    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)

    if len(real_set) == 0:
        raise RuntimeError("Real set is empty — check class_to_forget and dataset path.")
    if len(fake_set) == 0:
        raise RuntimeError("Fake set is empty — check generated image directory and filenames.")

    real_images = _denorm_to_uint8(torch.stack(real_set)).cpu()   # (N, C, H, W) uint8
    fake_images = _denorm_to_uint8(torch.stack(fake_set)).cpu()

    fid = FrechetInceptionDistance(feature=feature)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    fid_value = fid.compute().item()

    print(f"FID: {fid_value:.4f}  "
          f"(real={len(real_set)}, fake={len(fake_set)}, feature={feature})")

    fid.reset()
    del fid, real_images, fake_images
    return fid_value


def compute_fid_i2p(real_path: str,
                    path: str,
                    image_size: int,
                    feature: int = 64) -> float:
    """FID for I2P evaluation (no class filtering)."""
    real_set, fake_set = setup_fid_data_i2p(real_path, path, image_size)

    real_images = _denorm_to_uint8(torch.stack(real_set)).cpu()
    fake_images = _denorm_to_uint8(torch.stack(fake_set)).cpu()

    fid = FrechetInceptionDistance(feature=feature)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    fid_value = fid.compute().item()

    print(f"I2P FID: {fid_value:.4f}  "
          f"(real={len(real_set)}, fake={len(fake_set)})")

    fid.reset()
    del fid, real_images, fake_images
    return fid_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for unlearned model evaluation")

    parser.add_argument("--folder_path", type=str, required=False,default="/storage/s25017/MUNBa/SD/eval_scripts/CLASS/generated_images/diffusers-cls_0-MUNBa-method_full-lr_1e-05_E10_U963_topk15_pseudo_an_image-epoch_2.pt", help="Path to directory of generated images")
    parser.add_argument("--class_to_forget", type=int, nargs="+", required=False, default=[0],
                        help="One or more class indices to exclude (e.g. --class_to_forget 0 7)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--feature", type=int, default=2048,
                        choices=[64, 192, 768, 2048],
                        help="InceptionV3 feature layer for FID")

    # I2P mode
    parser.add_argument("--i2p", action="store_true", default=False,
                        help="Run I2P FID instead of Imagenette FID")
    parser.add_argument("--real_path", type=str, default=None,
                        help="Real image directory (required when --i2p is set)")

    args = parser.parse_args()

    if args.i2p:
        if args.real_path is None:
            raise ValueError("--real_path is required when --i2p is set")
        compute_fid_i2p(args.real_path, args.folder_path, args.image_size, args.feature)
    else:
        # Unwrap single-element list to int for cleaner downstream handling
        classes = args.class_to_forget if len(args.class_to_forget) > 1 else args.class_to_forget[0]
        compute_fid(classes, args.folder_path, args.image_size, args.feature)