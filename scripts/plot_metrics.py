#!/usr/bin/env python3
"""Plot validation curves and final accuracy bars from metrics.csv files.

Run:
    python scripts/plot_metrics.py --results-dir results --set short
    python scripts/plot_metrics.py --results-dir results --set long
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHORT_RUNS = [
    ("Baseline (ReLU + dropout)", "baseline_metrics.csv", "#8a3b2e"),
    ("tanh", "tanh_metrics.csv", "#2f5d7c"),
    ("No dropout", "no_dropout_metrics.csv", "#3f6b4f"),
]

LONG_RUNS = [
    ("Baseline (dropout 0.5)", "long_baseline_metrics.csv", "#8a3b2e"),
    ("dropout 0.25", "long_dropout025_metrics.csv", "#8a6a2f"),
    ("tanh (dropout 0.5)", "long_tanh_metrics.csv", "#2f5d7c"),
    ("No dropout", "long_no_dropout_metrics.csv", "#3f6b4f"),
]


def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_curve(results_dir: Path, runs, value: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(7.4, 4.3))
    for label, filename, color in runs:
        path = results_dir / filename
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        rows = read_rows(path)
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[value]) for row in rows]
        plt.plot(epochs, values, label=label, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / out_name, dpi=220)
    plt.close()


def plot_final_bar(results_dir: Path, runs, out_name: str) -> None:
    labels, top1, top5 = [], [], []
    for label, filename, _ in runs:
        path = results_dir / filename
        if not path.is_file():
            continue
        last = read_rows(path)[-1]
        labels.append(label)
        top1.append(float(last["val_top1"]))
        top5.append(float(last["val_top5"]))

    if not labels:
        print("no metrics files found")
        return

    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(7.6, 4.4))
    plt.bar([i - width / 2 for i in x], top1, width, label="Top-1", color="#8a3b2e")
    plt.bar([i + width / 2 for i in x], top5, width, label="Top-5", color="#2f5d7c")
    plt.xticks(list(x), labels, rotation=15, ha="right", fontsize=8)
    plt.ylabel("Validation accuracy (%)")
    plt.ylim(0, 75)
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / out_name, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--set", choices=["short", "long"], default="short")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.set == "long":
        runs = LONG_RUNS
        prefix = "long_"
    else:
        runs = SHORT_RUNS
        prefix = ""

    plot_curve(results_dir, runs, "val_top1", "Validation top-1 (%)", f"{prefix}val_top1.png")
    plot_curve(results_dir, runs, "val_loss", "Validation loss", f"{prefix}val_loss.png")
    plot_curve(results_dir, runs, "train_loss", "Training loss", f"{prefix}train_loss.png")
    plot_final_bar(results_dir, runs, f"{prefix}final_accuracy.png")

    print(f"Figures written to {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
