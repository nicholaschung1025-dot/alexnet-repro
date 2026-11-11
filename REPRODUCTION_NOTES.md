# Reproduction Notes

## Scope

- Paper: AlexNet (Krizhevsky, Sutskever & Hinton, NeurIPS 2012).
- What we reproduced: [architecture / training recipe / one ablation].
- What we did not reproduce: full ImageNet-1k, 90 epochs, two-GPU training,
  full PCA color augmentation, 10-patch test-time averaging.

## Setup

- Dataset: [Imagenette / ImageNet-100], [number of classes], [train/val sizes].
- Input size: 224×224.
- Hardware: 1× [RTX 4080 Super 16GB / RTX 3090 24GB].
- Epochs: [30].
- Batch size: [128].
- AMP: [on/off].
- Seed: [0].
- Git commit: [hash].

## Results

| Run | Top-1 accuracy | Top-5 accuracy | Epoch time | Notes |
|---|---:|---:|---:|---|
| Baseline | 14.06% | 35.04% | ~197s | ReLU + dropout + augmentation |
| tanh | 22.72% | 48.70% | ~215s | same recipe, tanh |
| No dropout | 29.18% | 56.64% | ~214s | dropout = 0 |
| Baseline, 90 ep | 28.34% | 56.06% | ~206s | 90-epoch extension |
| No dropout, 90 ep | 41.94% | 69.68% | ~208s | 90-epoch extension |

**Key finding:** both ablations outperformed the baseline. At this scale
(ImageNet-100, 30 epochs), dropout 0.5 appears too strong and causes
underfitting; tanh + dropout also beats ReLU + dropout. The 90-epoch extension
confirms the dropout result: no-dropout reaches 41.94% top-1 versus 28.34% for
the baseline, a 13.6-point gap. The 90-epoch baseline is still slightly below
the 30-epoch no-dropout run, so dropout underfits even with longer training.
This is a single-seed result and does not directly contradict the paper's full
ImageNet-1k claims, which used 1.2M images and 1,000 classes. It shows that the
paper's regularization and activation claims are regime-dependent.

## Deviations from the paper

- Dataset is [Imagenette / ImageNet-100], not ImageNet-1k.
- 30 epochs, not 90.
- One GPU, not two GTX 580s.
- Conv1 uses 224×224 with padding 2; the paper's text says 224, released code used 227.
- Conv2/4/5 use `groups=2` to emulate the two-GPU channel split; Conv1 uses
  `groups=1` because its input has only 3 channels.
- Fixed LR milestones, not manual validation-based decay.
- **Initialization:** Kaiming instead of the paper's Gaussian(0, 0.01) with bias 1.
  The paper initialization collapsed to uniform predictions (loss ≈ ln(100));
  Kaiming memorized a real 8-image batch to 100%.
- **Optimizer:** SGD with momentum 0.0 and no gradient clipping for stability;
  the paper used momentum 0.9. LR is 0.001 instead of 0.01.
- **AMP:** disabled; FP16 gradient overflow caused skipped optimizer steps.
- PCA color augmentation is [implemented / not implemented].

## Challenges

- [OOM at batch 128; reduced to 64.]
- [Data loading bottleneck; increased workers.]
- [Training instability; changed initialization / LR.]
- [Farm quota / session limit.]

## What we would do with more compute

- Train on ImageNet-100 for 90 epochs.
- Add PCA color augmentation.
- Run 2–3 seeds and report mean ± standard deviation.
- Reproduce the 10-patch test-time averaging.
- Compare against a width-matched baseline to separate architecture, data, and compute.

## Attribution

- Paper: Krizhevsky, Sutskever & Hinton (2012).
- Dataset: ImageNet / ILSVRC; Imagenette by fast.ai.
- Reused code: [list any borrowed code and its license].
- Our contributions: [model, training loop, configs, experiments, analysis].
