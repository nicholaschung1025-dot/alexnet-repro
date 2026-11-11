#!/usr/bin/env python3
"""Check AlexNet output shapes and parameter counts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import AlexNet, count_parameters  # noqa: E402


def main() -> None:
    x = torch.randn(2, 3, 224, 224)
    for num_classes in (10, 100, 1000):
        model = AlexNet(num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(
            f"num_classes={num_classes:4d} | "
            f"output={tuple(y.shape)} | "
            f"parameters={count_parameters(model):,}"
        )
        assert y.shape == (2, num_classes)


if __name__ == "__main__":
    main()
