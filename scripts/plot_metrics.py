#!/usr/bin/env python3
"""Plot validation curves and a final accuracy bar chart from metrics.csv files.

Expected files:
    results/baseline_metrics.csv
    results/tanh_metrics.csv
    results/no_dropout_metrics.csv

Run:
    pip install matplotlib
    python scripts/plot_metrics.py --results-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    ("Baseline (ReLU + dropout)", "baseline_metrics.csv", "#8a3b2e"),
    ("tanh", "tanh_metrics.csv", "#2f5d7c"),
    ("No dropout", "no_dropout_metrics.csv", "#3f6b4f"),
]


def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_curve(results_dir: Path, value: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, filename, color in RUNS:
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
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / out_name, dpi=220)
    plt.close()


def plot_final_bar(results_dir: Path) -> None:
    labels, top1, top5 = [], [], []
    for label, filename, _ in RUNS:
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
    plt.figure(figsize=(7.2, 4.2))
    plt.bar([i - width / 2 for i in x], top1, width, label="Top-1", color="#8a3b2e")
    plt.bar([i + width / 2 for i in x], top5, width, label="Top-5", color="#2f5d7c")
    plt.xticks(list(x), labels, rotation=10)
    plt.ylabel("Validation accuracy (%)")
    plt.ylim(0, 65)
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "final_accuracy.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    plot_curve(results_dir, "val_top1", "Validation top-1 (%)", "val_top1.png")
    plot_curve(results_dir, "val_loss", "Validation loss", "val_loss.png")
    plot_curve(results_dir, "train_loss", "Training loss", "train_loss.png")
    plot_final_bar(results_dir)

    print(f"Figures written to {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
