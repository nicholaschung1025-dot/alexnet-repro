#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate alexnet
cd "${SLURM_SUBMIT_DIR:-$HOME/alexnet-repro}"

for name in paper_baseline paper_tanh paper_dropout025 paper_no_dropout; do
    echo "== $name"
    python -u -m src.evaluate --config "configs/${name}.yaml" \
        --checkpoint "runs/${name}/best.pth" --ten-crop
done
