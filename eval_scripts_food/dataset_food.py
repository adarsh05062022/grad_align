# eval/dataset_food101.py
import os
import numpy as np
import torchvision.transforms as torch_transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import Food101

FOOD10_CLASSES = [
    "pizza",
    "sushi",
    "hamburger",
    "hot_and_sour_soup",
    "ice_cream",
    "chocolate_cake",
    "ramen",
    "steak",
    "tacos",
    "waffles",
]

FOOD10_CLASS_TO_IDX = {name: i for i, name in enumerate(FOOD10_CLASSES)}

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize((size, size), interpolation=interpolation),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


############################################
# INTERNAL: remap Food101 → 0-9 labels
############################################

def _get_food10_remapped_labels(dataset):
    """
    Returns list of int: 0-9 for our 10 classes, -1 for everything else.
    """
    return [
        FOOD10_CLASS_TO_IDX.get(dataset.classes[raw_label], -1)
        for raw_label in dataset._labels
    ]


############################################
# FOOD101 DATASET  (replaces CIFAR10Dataset)
############################################

class Food101Dataset(Dataset):
    def __init__(self, split="train", class_to_forget=None, transform=None):
        self.base = Food101(
            root="/storage/s25017/MUNBa/SD/datasets",
            split=split,
            download=False
        )
        self.remapped_labels = _get_food10_remapped_labels(self.base)

        # Keep only samples belonging to our 10 classes
        self.indices = [
            i for i, label in enumerate(self.remapped_labels)
            if label != -1
        ]

        self.class_to_forget = class_to_forget
        self.num_classes = len(FOOD10_CLASSES)   # 10
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base[real_idx]            # ignore raw Food101 label
        label = self.remapped_labels[real_idx]    # our 0-9 label

        # Randomize label for forgotten class (same logic as CIFAR version)
        if label == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)

        return image, label


############################################
# FAKE FOOD101  (replaces Fake_CIFAR)
# Reads generated images from a folder,
# filename format: {class_idx}_{anything}.png
############################################

class Fake_Food101(Dataset):
    def __init__(self, data_dir, class_to_forget, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Exclude images whose filename starts with the forget class index
        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".png") and not f.startswith(str(class_to_forget))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        class_idx = int(filename.split("/")[-1].split("_")[0])

        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path).convert("RGB")   # Food101 is always RGB but be safe

        if self.transform:
            image = self.transform(image)

        return image, class_idx


############################################
# FID DATA SETUP  (replaces setup_fid_data)
############################################

def setup_fid_data(class_to_forget, path, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Food101Dataset(split="train", transform=transform)

    real_set = []
    cnt = 0

    for image, label in train_set:
        if label != class_to_forget:
            real_set.append(image)
            cnt += 1
        if cnt >= 1000:
            break

    fake_set = Fake_Food101(path, class_to_forget, transform=transform)
    fake_set = [data[0] for data in fake_set]

    return real_set, fake_set