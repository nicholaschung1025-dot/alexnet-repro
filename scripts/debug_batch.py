#!/usr/bin/env python3
"""Check one real batch: image statistics, labels, and a few optimizer steps.

Run on a GPU node:
    python scripts/debug_batch.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataloaders  # noqa: E402
from src.model import AlexNet  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = 0
    train_loader, _, classes = build_dataloaders(cfg)

    images, labels = next(iter(train_loader))
    print("classes:", len(classes))
    print("images:", tuple(images.shape))
    print(
        "image stats: mean=%.4f std=%.4f min=%.4f max=%.4f"
        % (
            images.mean().item(),
            images.std().item(),
            images.min().item(),
            images.max().item(),
        )
    )
    print("labels[:20]:", labels[:20].tolist())
    print("unique labels in batch:", labels.unique().tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = AlexNet(
        num_classes=int(cfg["data"]["num_classes"]),
        dropout=float(cfg["model"].get("dropout", 0.5)),
        activation=str(cfg["model"].get("activation", "relu")),
        use_lrn=bool(cfg["model"].get("use_lrn", True)),
        init=str(cfg["model"].get("init", "paper")),
    ).to(device)
    model.train()

    images = images.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"].get("momentum", 0.9)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0005)),
    )
    criterion = torch.nn.CrossEntropyLoss()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        optimizer.step()
        print(
            f"step {step} | loss {loss.item():.4f} | grad_norm {grad_norm:.4f} | "
            f"logit_std {outputs.std().item():.4f}"
        )


if __name__ == "__main__":
    main()
