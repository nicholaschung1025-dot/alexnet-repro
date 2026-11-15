#!/usr/bin/env python3
"""Plot validation curves for the paper-recipe runs."""

from pathlib import Path

import csv

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("paper_baseline", "dropout 0.5", "#141414"),
    ("paper_dropout025", "dropout 0.25", "#5E5E5E"),
    ("paper_no_dropout", "no dropout", "#9A9A9A"),
    ("paper_tanh", "tanh", "#CFCFCF"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
for name, label, color in RUNS:
    with open(ROOT / "results" / f"{name}_metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    top1 = [float(row["val_top1"]) for row in rows]
    ax.plot(epochs, top1, label=label, color=color, linewidth=1.8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation top-1 (%)")
ax.set_xlim(1, 90)
ax.set_ylim(0, 75)
ax.grid(axis="y", color="#E3E5E7", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Paper recipe, ImageNet-100", fontsize=10, loc="left")

out = ROOT / "results" / "figures" / "paper_val_top1.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print(f"Saved {out}")
