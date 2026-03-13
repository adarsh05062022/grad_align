# https://huggingface.co/docs/diffusers/conceptual/evaluation
#compute-fid.py
import argparse

import torch
from dataset import setup_fid_data, setup_fid_data_i2p
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(class_to_forget, path, image_size):
    fid = FrechetInceptionDistance(feature=64)
    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)
    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP
    fid_value = fid.compute()
    print(fid_value)
    fid.reset()
    del fid
    return fid_value.item() # doctest: +SKIP


def compute_fid_i2p(real_path, path, image_size):
    fid = FrechetInceptionDistance(feature=64)
    real_set, fake_set = setup_fid_data_i2p(real_path, path, image_size, interpolation="bicubic")
    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP
    print(fid.compute())  # doctest: +SKIP



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=False, default="/storage/s25017/MUNBa/SD/eval_scripts/CLASS/generated_images/diffusers-cls_7-MUNBa-method_full-lr_1e-05_E5_U931_masked_nash_topk15_frequent_mask.pt")
    parser.add_argument("--class_to_forget", type=int, nargs="+", required=False, default=[0],
                        help="One or more class indices to exclude (e.g. --class_to_forget 0 7)")
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    # parser.add_argument("--real_path", help="path of images", type=str, required=True)
    args = parser.parse_args()

    # image_size = args.image_size
    # path = args.folder_path
    # real_path = args.real_path
    # compute_fid_i2p(real_path, path, image_size)

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print(class_to_forget)
    compute_fid(class_to_forget, path, image_size)
