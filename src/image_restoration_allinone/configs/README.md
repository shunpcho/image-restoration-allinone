# Config Module

This directory contains the configuration definitions and loader logic used during training and evaluation of the image restoration model.

Its main responsibilities are:

- Defining default settings
- Loading configuration from YAML and CLI options
- Converting values into typed configuration objects

The config system handles both default values based on `fvcore.common.config.CfgNode` and a typed `Config` dataclass-based object.

---

## 1. Directory roles

- `default.py`
  - Defines the default configuration as a `CfgNode`.
  - Exposes `get_default_cfg()`, which provides the shared default values used across the project.

- `config_class.py`
  - Defines `DataConfig`, `ModelConfig`, `LossConfig`, `TrainConfig`, `LoggingConfig`, and `Config`.
  - Provides `config_from_cfg_node()`, which converts a `CfgNode` into typed dataclass objects.

- `parser.py`
  - Parses CLI arguments and loads configuration.
  - Accepts a YAML file via `--config` and allows additional overrides in `key=value` form.

- `generate_config.py`
  - Contains helper logic that automatically converts architecture-specific arguments into dataclasses.
  - Generates the appropriate config class for each architecture registered in `MODEL_REGISTRY`.

- `__init__.py`
  - Exposes the configuration classes for convenient imports from outside the module.

---

## 2. Configuration hierarchy

The project configuration is split into five major sections. The `model` section defines the architecture and its parameters.

### `model`

This section defines the model architecture and architecture-specific parameters.

`ModelConfig` manages `arch_name` and `parameters` together and validates the values to ensure invalid settings are rejected for the selected model.

## 3. Actual config loading flow

Config values are typically loaded in the following order:

1. `get_default_cfg()` in `default.py` creates the default values
2. A YAML config is loaded via `--config`
3. Extra CLI overrides are applied
4. `config_from_cfg_node()` converts the `CfgNode` into a `Config` object

`parser.py` performs the following logic:

```python
cfg = get_default_cfg()

if args.config is not None:
    cfg.merge_from_file(args.config)
if args.options:
    cfg.merge_from_list(args.options)

cfg.freeze()
return config_from_cfg_node(cfg)
```

This design satisfies all of the following simultaneously:

- It provides default values
- It allows settings to be changed via YAML
- It supports dynamic overrides from the CLI
- It exposes a type-safe `Config` object for application code

---

## 4. Example YAML configuration

For example, a simple configuration file can be written as follows:

```yaml
data:
  data_root: data/sample
  patch_size: 256
  val_ratio: 0.1

model:
  arch_name: NAFNet
  nafnet:
    width: 32
    num_enc_blks: [1, 1, 1, 28]
    middle_blk_num: 1
    num_dec_blks: [1, 1, 1, 1]

loss:
  losses:
    mse: 1.0
    ssim: 0.1

train:
  output_dir: results/default
  batch_size: 8
  epochs: 10
  lr: 0.001

logging:
  experiment_name: image_restoration_allinone
```

It can be used at runtime as follows:

```bash
uv run train --config config_yaml/config.yaml
```

You can also override specific values from the CLI:

```bash
uv run train --config config_yaml/config.yaml train.epochs=20 train.batch_size=16
```

---

## 5. Python usage examples

```python
from image_restoration_allinone.configs import Config, config_from_cfg_node
from image_restoration_allinone.configs.default import get_default_cfg

cfg = get_default_cfg()
cfg.merge_from_file("config_yaml/config.yaml")
config: Config = config_from_cfg_node(cfg)

print(config.data.data_root)
print(config.model.arch_name)
print(config.train.epochs)
print(config.loss.losses)
```

You can also construct config objects directly:

```python
from image_restoration_allinone.configs import DataConfig, TrainConfig, LossConfig, ModelConfig, Config

config = Config(
    data=DataConfig(data_root="data/sample"),
    model=ModelConfig(arch_name="NAFNet"),
    loss=LossConfig(losses={"mse": 1.0}),
    train=TrainConfig(epochs=10),
)
```

---

## 6. Design notes

The purpose of this configuration module is as follows:

- `CfgNode` makes it easy to work with YAML and CLI inputs
- `dataclass` objects add strong typing and validation
- Architecture-specific parameters are separated to keep the code maintainable
- Responsibilities such as `loss`, `train`, and `logging` are isolated for easier extension

Additionally, `ModelConfig` enforces constraints so invalid settings are not mixed in based on `arch_name`, helping reduce configuration errors in practice.

---

## 7. Summary

`src/image_restoration_allinone/configs` is the entry point for configuration in this project.

It handles four main responsibilities:

- Defining default values
- Overriding values via YAML and CLI
- Creating typed configuration data
- Organizing model, loss, training, and logging settings

These settings are used by other training and inference modules throughout the project.

For more details, refer to the following files:

- `default.py`
- `config_class.py`
- `parser.py`
- `config_yaml/config.yaml`
