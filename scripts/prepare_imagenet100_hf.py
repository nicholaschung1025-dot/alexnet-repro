#!/usr/bin/env python3
"""Convert clane9/imagenet-100 from Hugging Face into ImageFolder format.

Output:
    <output>/train/class_000/*.jpg
    <output>/val/class_000/*.jpg

The Hugging Face version resizes images to 160 px on the shorter side. We
document that deviation in REPRODUCTION_NOTES.md; RandomResizedCrop(224) will
upscale them during training.

Usage:
    pip install datasets huggingface_hub tqdm pillow
    python scripts/prepare_imagenet100_hf.py --output ~/data/imagenet100

For a quick subset:
    python scripts/prepare_imagenet100_hf.py --output ~/data/imagenet100_small --max-per-class 100
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="~/data/imagenet100",
        help="Destination ImageFolder root.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for a quick subset.",
    )
    parser.add_argument(
        "--dataset",
        default="clane9/imagenet-100",
        help="Hugging Face dataset id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    dataset = load_dataset(args.dataset, trust_remote_code=True)

    split_map = {"train": "train", "validation": "val"}
    for source_split, target_split in split_map.items():
        if source_split not in dataset:
            raise KeyError(
                f"Split {source_split!r} not found. Available: {list(dataset.keys())}"
            )
        split = dataset[source_split]
        counts: Counter[int] = Counter()
        written = 0
        skipped = 0

        for index, example in enumerate(
            tqdm(split, desc=f"{source_split} -> {target_split}")
        ):
            label = int(example["label"])
            if args.max_per_class is not None and counts[label] >= args.max_per_class:
                skipped += 1
                continue

            class_dir = output / target_split / f"class_{label:03d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(class_dir / f"{index:06d}.jpg", quality=95)

            counts[label] += 1
            written += 1

        print(
            f"{target_split}: wrote {written:,} images across "
            f"{len(counts)} classes; skipped {skipped:,}."
        )

    print(f"Done: {output}")
    print("Expected: 100 class folders under train/ and val/.")


if __name__ == "__main__":
    main()
