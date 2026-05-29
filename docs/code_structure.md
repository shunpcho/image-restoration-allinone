# Code Structure

```
  src/{project_name}
  |-- configs      # Configuration files for datasets, models, logging, and experiments.
  |-- data         # Data loading, augmentation, preprocessing, and dataset utilities.
  |-- models       # Model definitions, layers, and model-building modules.
  |-- utils        # Shared utilities (training helpers, logging, losses, and common functions).
  |-- train.py     # Training entry point.
  `-- inference.py # Inference/evaluation entry point.
```

## Description

- `configs`: Centralized configuration files to control data, model, runtime, and logging behavior.
- `data`: Components for reading datasets and preparing inputs for training and inference.
- `models`: Core network architectures and reusable model components.
- `utils`: Common helper modules used across the project.
- `train.py`: Script to launch model training.
- `inference.py`: Script to run prediction or evaluation with trained weights.