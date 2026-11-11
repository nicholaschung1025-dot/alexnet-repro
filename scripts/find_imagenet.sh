#!/usr/bin/env bash
# Search common shared paths on the HKU GPU farm for ImageNet / ImageNet-100.
set -euo pipefail

echo "Searching for ImageNet-related directories..."
echo "This may take a minute; it only checks common shared paths."
echo

for root in /share /data /datasets /mnt /scratch /userhome; do
  if [ ! -d "$root" ]; then
    continue
  fi
  echo "== $root =="
  find "$root" -maxdepth 4 \
    \( -iname '*imagenet*' -o -iname '*ilsvrc*' -o -iname '*imagenet100*' \) \
    2>/dev/null | head -30
  echo
done

echo "If nothing appears:"
echo "  1. Ask your TA/instructor whether the course has a shared ImageNet path."
echo "  2. Use scripts/prepare_imagenet100_hf.py (Hugging Face, ~8.4 GB)."
echo "  3. Fall back to Imagenette if the deadline is close."
