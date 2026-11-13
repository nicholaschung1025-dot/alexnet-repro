"""Evaluate a trained AlexNet checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn

from src.data import build_dataloaders
from src.engine import evaluate
from src.model import AlexNet
from src.utils import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AlexNet.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--ten-crop",
        action="store_true",
        help="Override data.ten_crop for the paper's 10-crop evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.ten_crop:
        cfg["data"]["ten_crop"] = True
    device = get_device()

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model = AlexNet(
        num_classes=int(data_cfg["num_classes"]),
        dropout=float(model_cfg.get("dropout", 0.5)),
        activation=str(model_cfg.get("activation", "relu")),
        use_lrn=bool(model_cfg.get("use_lrn", True)),
        init=str(model_cfg.get("init", "paper")),
        paper_bias=float(model_cfg.get("paper_bias", 1.0)),
    ).to(device)

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(cfg["output"]["dir"]).expanduser() / "best.pth"
    )
    load_checkpoint(checkpoint, model, device=device)

    _, val_loader, classes = build_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        amp=bool(cfg["train"].get("amp", True)),
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Classes: {classes}")
    print(
        f"val loss {metrics['loss']:.4f} | "
        f"top-1 {metrics['top1']:.2f}% | "
        f"top-5 {metrics['top5']:.2f}%"
    )


if __name__ == "__main__":
    main()
