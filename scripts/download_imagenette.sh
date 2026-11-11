#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-$HOME/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d "imagenette2-320" ]; then
  wget -c https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
  tar -xzf imagenette2-320.tgz
fi

echo "Dataset ready: $DATA_DIR/imagenette2-320"
