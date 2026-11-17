#!/usr/bin/env python3
"""Colored validation curves for the paper-recipe runs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("paper_baseline", "dropout 0.5", "#111111", "-"),
    ("paper_dropout025", "dropout 0.25", "#0072B2", "-"),
    ("paper_no_dropout", "no dropout", "#E69F00", "-"),
    ("paper_tanh", "tanh", "#D55E00", "-"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
ends = []
for name, label, color, style in RUNS:
    with open(ROOT / "results" / f"{name}_metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    top1 = [float(row["val_top1"]) for row in rows]
    ax.plot(epochs, top1, color=color, linestyle=style, linewidth=2.2,
            solid_capstyle="round")
    ax.plot(epochs[-1], top1[-1], "o", color=color, markersize=5)
    ends.append((top1[-1], label, color))

for value, label, color in ends:
    ax.annotate(f"{label}  {value:.1f}", xy=(90, value), xytext=(92.5, value),
                va="center", fontsize=8.5, color=color, fontweight="bold")

ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("Validation top-1 (%)", fontsize=9)
ax.set_xlim(1, 90)
ax.set_ylim(0, 78)
ax.grid(axis="y", color="#E3E5E7", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5, colors="#3D3D3D")

out = ROOT / "results" / "figures" / "paper_val_top1_color.png"
fig.tight_layout()
fig.savefig(out, facecolor="white")
print(f"Saved {out}")
