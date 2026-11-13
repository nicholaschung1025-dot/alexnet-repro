#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate alexnet
cd "${SLURM_SUBMIT_DIR:-$HOME/alexnet-repro}"

CONFIG="${1:-configs/baseline.yaml}"
echo "Config: $CONFIG"
echo "Host: $(hostname)"
nvidia-smi
python -u -m src.train --config "$CONFIG"
