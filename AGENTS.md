# Agent Instructions (Copilot / AI assistants)

## Instructions

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a proven foundation.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

## Overview

This is a template repository providing guidelines, best practices, and conventions for Python projects working with AI assistants. It defines agent instructions, coding standards, project structure, and dependency management practices.

## 0) Where to find coding rules

- Coding rules (style, typing, testing, tooling) are defined in:
  - `.github/copilot-instructions.md`
- Consult it before making stylistic decisions.

## 1) Priority / scope

- Follow this file first.
- Then follow `.github/copilot-instructions.md`.
- Then follow other repository docs (`README.md`, `CONTRIBUTING.md`, `docs/*`).
- If instructions conflict, ask for clarification rather than guessing.

## 2) Repository structure (must follow)

Expected at project root:

- `pyproject.toml` (required; single source of truth)
- `src/<package_name>/` (package code; keep `src/` layout consistent)
- `tests/`
- `docs/`

## 3) Dependency management (uv required)

- Dependency management MUST use **uv**.
- Prefer `uv run ...` to execute tools in the managed environment.

## 4) What to include in proposals/patches

- Exact file paths to create/edit.
- Final code (not only pseudocode).
- Brief rationale.
- How to run formatting/lint/type-check/tests (uv-based commands).

## 5) Computer Vision Framework

This repository includes documentation for building a supervised learning **Image Restoration** framework using PyTorch, with experiment tracking via MLflow.

### Custom Agents (`.github/agents/`)

Each agent lives in `.github/agents/` as a `.agent.md` file with YAML frontmatter and instructions.

| File                           | Role                                                                              |
| ------------------------------ | --------------------------------------------------------------------------------- |
| `cv-data-agent.agent.md`       | Pair dataset loading, synthetic degradation pipelines, and patch-based DataLoader |
| `cv-training-agent.agent.md`   | Model training with pixel/perceptual/SSIM loss, MLflow experiment tracking        |
| `cv-evaluation-agent.agent.md` | PSNR / SSIM / LPIPS evaluation, side-by-side visualization, MLflow logging        |

### Skills (`.github/skills/`)

Each skill lives in its own subdirectory with a `SKILL.md` file containing YAML frontmatter and instructions.

| Directory            | Task                                                                     |
| -------------------- | ------------------------------------------------------------------------ |
| `image-denoising/`   | Gaussian / Poisson / real-world noise removal (DnCNN, NAFNet, Restormer) |
| `super-resolution/`  | Single image super-resolution ×2/×3/×4 (EDSR, SwinIR, Real-ESRGAN)       |
| `image-deblurring/`  | Motion / defocus blur removal (NAFNet, MPRNet, DeblurGAN-v2)             |
| `data-augmentation/` | Pair-consistent augmentation pipelines using albumentations              |

## 6) Key resources

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) - Referenced for style conventions
- [Ruff](https://docs.astral.sh/ruff/) - Fast Python linter and formatter
- [Pyright](https://github.com/microsoft/pyright) - Static type checker
- [UV](https://github.com/astral-sh/uv) - Python package manager
