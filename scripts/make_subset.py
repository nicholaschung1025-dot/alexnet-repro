#!/usr/bin/env python3
"""Create a smaller ImageFolder subset using symlinks.

Example:
    python scripts/make_subset.py \
        --source ~/data/imagenet100 \
        --output ~/data/imagenet100_10k \
        --train-per-class 100 --val-per-class 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="~/data/imagenet100")
    parser.add_argument("--output", default="~/data/imagenet100_10k")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()

    for split, limit in (("train", args.train_per_class), ("val", args.val_per_class)):
        src_split = source / split
        if not src_split.is_dir():
            raise FileNotFoundError(src_split)
        for class_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
            files = sorted(class_dir.glob("*.jpg"))[:limit]
            dst_dir = output / split / class_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                link = dst_dir / file.name
                if not link.exists():
                    link.symlink_to(file)
        count = len(list((output / split).glob("*/*.jpg")))
        print(f"{split}: {count:,} images")

    print(f"Subset ready: {output}")


if __name__ == "__main__":
    main()
