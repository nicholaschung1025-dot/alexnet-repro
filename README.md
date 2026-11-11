# AlexNet Reproduction (COMP3314)

A from-scratch PyTorch implementation of AlexNet, with a partial reproduction on
**Imagenette** or **ImageNet-100**. This is a course project, not a claim that we
matched the paper's ImageNet-1k numbers.

**Paper:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton.
“ImageNet Classification with Deep Convolutional Neural Networks.”
*NeurIPS 2012*.

## Scope

- AlexNet implemented from scratch (no `torchvision.models.alexnet`).
- Five convolution layers, three fully connected layers, ReLU, LRN,
  overlapping max pooling, dropout.
- Partial reproduction on a smaller dataset, not full ImageNet-1k.
- Baseline plus controlled ablations: ReLU vs. tanh, dropout, augmentation.
- One GPU; the original two-GPU split is emulated with `groups=2` where valid.

## Results

| Run | Dataset | Epochs | Top-1 accuracy | Top-5 accuracy | Notes |
|---|---|---|---:|---:|---|
| Paper (ILSVRC-2010) | ImageNet-1k | 90 | 62.5% (37.5% error) | 83.0% (17.0% error) | 2× GTX 580 |
| Ours: baseline | ImageNet-100 | 30 | **14.06%** | **35.04%** | ReLU + dropout, 1× RTX 4080 Super |
| Ours: tanh | ImageNet-100 | 30 | **22.72%** | **48.70%** | same recipe, tanh |
| Ours: no dropout | ImageNet-100 | 30 | **29.18%** | **56.64%** | dropout = 0 |
| Ours: baseline, 90 ep | ImageNet-100 | 90 | **28.34%** | **56.06%** | dropout 0.5 |
| Ours: tanh, 90 ep | ImageNet-100 | 90 | **33.12%** | **60.70%** | tanh + dropout 0.5 |
| Ours: dropout 0.25, 90 ep | ImageNet-100 | 90 | **35.68%** | **63.98%** | ReLU + dropout 0.25 |
| Ours: no dropout, 90 ep | ImageNet-100 | 90 | **41.94%** | **69.68%** | dropout 0.0 |

Paper numbers are accuracy derived from the reported error rates. **Do not compare
absolute numbers across different datasets.** The useful comparison is the
**relative trend** between our runs.

At 90 epochs, dropout shows a clear **dose-response**:

```text
dropout 0.5  → 28.34%
dropout 0.25 → 35.68%
dropout 0.0  → 41.94%
```

Less dropout is better at this scale. The 90-epoch baseline (28.34%) is still
slightly below the **30-epoch** no-dropout run (29.18%), so dropout 0.5 underfits
even with a longer schedule. tanh + dropout 0.5 (33.12%) also beats ReLU +
dropout 0.5 (28.34%), but remains below the lower-dropout runs.

## Repository layout

```text
alexnet-repro/
├── README.md
├── REPRODUCTION_NOTES.md
├── requirements.txt
├── LICENSE
├── train.slurm
├── configs/
│   ├── baseline.yaml       # ImageNet-100, 100 classes
│   ├── imagenet100.yaml    # alias/example for ImageNet-100
│   ├── imagenette.yaml     # fallback: Imagenette, 10 classes
│   ├── overfit.yaml
│   ├── smoke.yaml
│   ├── tanh.yaml
│   ├── no_dropout.yaml
│   └── no_aug.yaml
├── src/
│   ├── model.py
│   ├── data.py
│   ├── engine.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── scripts/
│   ├── check_model.py
│   ├── download_imagenette.sh
│   ├── find_imagenet.sh
│   └── prepare_imagenet100_hf.py
├── logs/
├── results/
└── checkpoints/
```

## Setup

On the GPU node:

```bash
conda create -n alexnet python=3.11 -y
conda activate alexnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

The default `configs/baseline.yaml` is set to **ImageNet-100**. Use
`configs/imagenette.yaml` if you need the Imagenette fallback.

### Option A: Imagenette (fallback for a tight deadline)

```bash
bash scripts/download_imagenette.sh
```

Expected layout:

```text
~/data/imagenette2-320/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Use the provided fallback config:

```bash
python -m src.train --config configs/imagenette.yaml
```

It overrides `data.root` to `~/data/imagenette2-320` and `num_classes` to `10`.

### Option B: ImageNet-100

First check whether the farm already has it:

```bash
bash scripts/find_imagenet.sh
```

Also ask your TA/instructor whether the course has a shared ImageNet path.

If not, download the Hugging Face version and convert it to ImageFolder:

```bash
pip install datasets huggingface_hub tqdm

python scripts/prepare_imagenet100_hf.py --output ~/data/imagenet100
```

Expected layout:

```text
~/data/imagenet100/
├── train/
│   ├── class_000/
│   └── ... 100 class folders
└── val/
    ├── class_000/
    └── ... 100 class folders
```

Validate:

```bash
ls ~/data/imagenet100/train | wc -l   # should be 100
find ~/data/imagenet100/train -type f | wc -l   # about 126k
```

This is already the default in `configs/baseline.yaml`:

```yaml
data:
  root: ~/data/imagenet100
  num_classes: 100
```

You can also run the explicit alias:

```bash
python -m src.train --config configs/imagenet100.yaml
```

**Note:** the Hugging Face version resizes images to **160 px on the shorter
side**. `RandomResizedCrop(224)` upscales them; document this deviation in
`REPRODUCTION_NOTES.md`. For a quick subset, add `--max-per-class 100`.

Do not upload full ImageNet to the GPU farm.

## Quick checks

```bash
# Model shape and parameter count
python scripts/check_model.py

# One-batch overfit test (does not need a dataset)
python -m src.train --config configs/overfit.yaml --overfit-one-batch --overfit-steps 200

# Two real batches
python -m src.train --config configs/smoke.yaml

# Timing / stability check: 50 train + 50 val batches
python -m src.train --config configs/timing.yaml
python -m src.train --config configs/timing_amp.yaml
```

## Training

```bash
# Baseline
python -m src.train --config configs/baseline.yaml

# Ablations
python -m src.train --config configs/tanh.yaml
python -m src.train --config configs/no_dropout.yaml
```

Or submit as a batch job from the gateway:

```bash
sbatch train.slurm configs/baseline.yaml
sbatch train.slurm configs/tanh.yaml
sbatch train.slurm configs/no_dropout.yaml

squeue -u $USER
tail -f logs/train-<job_id>.out
```

Checkpoints and metrics go to `runs/<name>/`:

```text
runs/baseline/
├── config.yaml
├── metrics.csv
├── last.pth
└── best.pth
```

Resume:

```bash
python -m src.train --config configs/baseline.yaml --resume
```

## Evaluation

```bash
python -m src.evaluate --config configs/baseline.yaml --checkpoint runs/baseline/best.pth
```

## Model details

| Layer | Configuration | Output (224 input) |
|---|---|---|
| Conv1 | 96 × 11×11, stride 4, padding 2, groups=1 | 96 × 55 × 55 |
| LRN1 | size 5, alpha 1e-4, beta 0.75, k 2 | 96 × 55 × 55 |
| MaxPool1 | 3×3, stride 2 | 96 × 27 × 27 |
| Conv2 | 256 × 5×5, padding 2, groups=2 | 256 × 27 × 27 |
| LRN2 | same as LRN1 | 256 × 27 × 27 |
| MaxPool2 | 3×3, stride 2 | 256 × 13 × 13 |
| Conv3 | 384 × 3×3, padding 1, groups=1 | 384 × 13 × 13 |
| Conv4 | 384 × 3×3, padding 1, groups=2 | 384 × 13 × 13 |
| Conv5 | 256 × 3×3, padding 1, groups=2 | 256 × 13 × 13 |
| MaxPool3 | 3×3, stride 2 | 256 × 6 × 6 |
| FC1 | 9216 → 4096, ReLU, dropout 0.5 | 4096 |
| FC2 | 4096 → 4096, ReLU, dropout 0.5 | 4096 |
| FC3 | 4096 → num_classes | num_classes |

Parameter count is about **57M for 100 classes** and **56.9M for 10 classes**,
depending on the classifier size.

## Differences from the paper

- **Dataset:** Imagenette (10 classes) or ImageNet-100, not ImageNet-1k.
- **Epochs:** 30, not 90.
- **Hardware:** one GPU, not two GTX 580s.
- **Input:** 224×224 with padding 2 in Conv1, which reproduces the paper's 55×55
  output; the paper's text says 224, while its released code used 227.
- **Two-GPU split:** emulated with `groups=2` for Conv2/4/5; Conv1 uses `groups=1`
  because its input has only 3 channels. Conv3 mixes both streams.
- **Learning-rate schedule:** fixed milestones at 50% and 75% of training, not
  manual validation-based decay.
- **Augmentation:** RandomResizedCrop + horizontal flip; PCA color augmentation
  is not implemented by default.
- **Initialization:** Kaiming instead of the paper's Gaussian(0, 0.01) with
  bias 1. The paper initialization collapsed to uniform predictions in our
  runs; Kaiming memorized a real batch 100%.
- **Optimizer:** SGD with momentum 0.0 and no gradient clipping for stability;
  the paper used momentum 0.9. Both are documented deviations.
- **AMP:** disabled for the baseline to avoid FP16 gradient-overflow skips.

## Attribution

- Paper: Krizhevsky, Sutskever & Hinton (2012).
- Dataset: ImageNet / ILSVRC; Imagenette by fast.ai.
- PyTorch and torchvision for data loading and building blocks.
- This implementation was written for the COMP3314 course project. If you reuse
  code, keep the attribution and license.

## License

MIT. See `LICENSE`.
