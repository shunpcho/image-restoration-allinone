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

## Dataset Structure

See [`docs/data_structure.md`](docs/data_structure.md) for supported dataset layouts.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```
