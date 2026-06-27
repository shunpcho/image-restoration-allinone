# image-restoration-allinone

All-in-one image restoration model that handles multiple degradation types (blur, low-light, rain, etc.) using a single NAFNet model trained on paired data.

## Features

- **Single model** for multiple degradation types — no task-specific switching required
- **MSE loss by default**, easily extendable with L1, Charbonnier, SSIM, Perceptual loss
- **No synthetic degradation** — uses real paired (degraded, clean) datasets
- **Validation metrics**: loss, MSE, PSNR, SSIM
- **MLflow** experiment tracking

## Quickstart

```bash
# Install dependencies
uv sync --all-groups

# Train
uv run train \
  --data-root /path/to/dataset \
  --output-dir results \
  --losses "mse:1.0"

# Train with multiple losses
uv run train \
  --data-root /path/to/dataset \
  --losses "mse:1.0,ssim:0.1,charbonnier:0.5"

# Inference / evaluation
uv run python -m image_restoration_allinone.inference \
  --checkpoint results/final.pth \
  --data-root /path/to/dataset \
  --save-images
```
## Dataset

### Dataset Structure

See [`docs/data_structure.md`](docs/data_structure.md) for supported dataset layouts.

### Data LoViF

This is datasets of [LoViF](https://lovif-cvpr2026-workshop.github.io/).  
**LoViF** is a workshop in CVPR2026.  
It has 5 copmetitions.
This datasets is from one of the cpmetitions, "[The First Challenge on Real-World All-in-One Image Restoration](https://www.codabench.org/competitions/13251/)" from [FoundIR](https://github.com/House-Leo/FoundIR).

### Files

- `Train.zip`: Train dataset include in several types of degration.
- `Testset_GT.zip`: Test dataset of ground truth.
- `Testset_LQ.zip`: Test dataset of low quality.
- `Val_LQ.zip`: Validate dataset of low qality, but it hasn't ground truth.
- `Test_LQ.zip`: Test dataset of low qality, but it hasn't ground truth.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```
