#!/bin/bash
# Submit the four paper-recipe runs: same init, optimiser, augmentation, eval.
# Usage: ./submit_paper.sh
set -euo pipefail

for name in paper_baseline paper_tanh paper_dropout025 paper_no_dropout; do
    sbatch -J "$name" train.slurm "configs/${name}.yaml"
done
