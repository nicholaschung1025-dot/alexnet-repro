"""Dataset and DataLoader construction for ImageFolder-style datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transform(input_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    resize_size = int(round(input_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _val_transform(input_size: int) -> transforms.Compose:
    resize_size = int(round(input_size * 256 / 224))
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
    train_dataset = datasets.ImageFolder(
        train_dir, transform=_train_transform(input_size, augment)
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=_val_transform(input_size))

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
