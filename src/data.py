"""Dataset and DataLoader construction for ImageFolder-style datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageNet RGB PCA basis used by the paper's colour augmentation. Applied to
# tensors in [0, 1], as in the reference AlexNet implementations.
PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
PCA_EIGVAL = [0.2175, 0.0188, 0.0045]


class PCAColourJitter:
    """The paper's PCA colour augmentation on a [0, 1] CHW tensor."""

    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = sigma
        self.eigvec = torch.tensor(PCA_EIGVEC, dtype=torch.float32)
        self.eigval = torch.tensor(PCA_EIGVAL, dtype=torch.float32)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        alpha = torch.normal(0.0, self.sigma, (3,))
        shift = self.eigvec @ (alpha * self.eigval)
        return (image + shift.view(3, 1, 1)).clamp_(0.0, 1.0)


def _train_transform(input_size: int, augment: bool, paper_crop: bool = False,
                     pca_lighting: bool = False) -> transforms.Compose:
    if augment:
        steps = []
        if paper_crop:
            steps.append(transforms.Resize(int(round(input_size * 256 / 224))))
            steps.append(transforms.RandomCrop(input_size))
        else:
            steps.append(transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)))
        steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.ToTensor())
        if pca_lighting:
            steps.append(PCAColourJitter())
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(steps)
    resize_size = int(round(input_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _stack_crops(crops) -> torch.Tensor:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return torch.stack(
        [normalize(TF.to_tensor(crop)) for crop in crops]
    )


def _val_transform(input_size: int, ten_crop: bool = False) -> transforms.Compose:
    resize_size = int(round(input_size * 256 / 224))
    if ten_crop:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.TenCrop(input_size),
                transforms.Lambda(_stack_crops),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"]).expanduser()
    train_dir = root / data_cfg.get("train_dir", "train")
    val_dir = root / data_cfg.get("val_dir", "val")

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Expected ImageFolder layout: <root>/train/<class>/*.JPEG"
        )
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Expected ImageFolder layout: <root>/val/<class>/*.JPEG"
        )

    input_size = int(data_cfg.get("input_size", 224))
    augment = bool(data_cfg.get("augment", True))
    paper_crop = bool(data_cfg.get("paper_crop", False))
    pca_lighting = bool(data_cfg.get("pca_lighting", False))
    ten_crop = bool(data_cfg.get("ten_crop", False))
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=_train_transform(input_size, augment, paper_crop, pca_lighting),
    )
    val_dataset = datasets.ImageFolder(
        val_dir, transform=_val_transform(input_size, ten_crop)
    )

    expected_classes = int(data_cfg["num_classes"])
    if len(train_dataset.classes) != expected_classes:
        raise ValueError(
            f"Found {len(train_dataset.classes)} classes in {train_dir}, "
            f"but data.num_classes={expected_classes}."
        )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation class folders differ. "
            f"train={train_dataset.classes[:5]}... val={val_dataset.classes[:5]}..."
        )

    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    batch_size = int(data_cfg.get("batch_size", 128))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_dataset.classes
