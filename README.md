# image-restoration-allinone

All-in-one image restoration model that handles multiple degradation types (blur, low-light, rain, etc.) using a single NAFNet model trained on paired data.

## Features

- **Supervised training** for image restoration using paired degraded/clean image datasets.
- **Validate score** for loss, MSE, PSNR, SSIM, and LoViF scores during training.
- **Flexible configuration** with YAML/CLI overrides and typed config objects for data, model, loss, training, and logging. See [config documentation](src/image_restoration_allinone/configs/README.md) for configuration details.
- **MLflow integration** for experiment tracking, metric logging, and result comparison.

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

Two dataset variants are available:

- The [sample dataset](data/sample) is a minimal dataset for debugging.
- The [training dataset](data/train) is managed with DVC and is used for LoViF experiments.

- Download the training dataset with DVC:

```bash
dvc pull train
```

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

### Adding a new model

To add a new restoration model, register it in the model registry and expose its constructor arguments through the config system.

1. Create a new model file under `src/image_restoration_allinone/models/`.
2. Implement a `torch.nn.Module` subclass.
3. Register the model with `MODEL_REGISTRY`.
4. Optionally define the model-specific config values in `default.py`.
5. Use `arch_name` in the config to select the model.

Example:

```python
# src/image_restoration_allinone/models/my_model.py
from torch import nn
from image_restoration_allinone.models.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MyModel(nn.Module):
    def __init__(self, width: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks

    def forward(self, x):
        return x
```

Then add a config entry like this:

```python
cfg.model.arch_name = "MyModel"
cfg.model.my_model = CfgNode()
cfg.model.my_model.width = 32
cfg.model.my_model.num_blocks = 4
```

The registry-based design means the project can automatically build the model from the selected architecture name. In other words, once the model is registered and its constructor signature matches the config fields, it can be used by the training pipeline without extra custom wiring.

When you want to use it from the CLI or YAML config, set:

```yaml
model:
  arch_name: MyModel
  my_model:
    width: 32
    num_blocks: 4
```
