---
applyTo: "**/*.py"
description: "Python coding standards for maintainable, typed, testable, and readable code."
---

# Python Coding Guidelines

These guidelines are for AI coding agents working on Python files. They complement, rather than duplicate, checks enforced by Ruff, Pyright, Ty, pytest, and project configuration files. Follow the configuration in `pyproject.toml`.

## Priority Order

When rules appear to conflict, follow this order:

1. Correctness and safety
2. Existing project conventions
3. Public API compatibility
4. Type safety
5. Readability and maintainability
6. Performance
7. Brevity

Do not introduce a large abstraction merely to satisfy a stylistic preference.

## Version and Tooling Assumptions

- Follow the Python version and dependencies declared in `pyproject.toml`.
- Prefer modern Python syntax supported by the declared target version.
- Keep code compatible with Ruff formatting and linting.
- Keep code type-checkable by Pyright and Ty.
- Do not bypass tools with broad ignores. Use narrow, justified suppressions only when necessary.
- Do not shadow standard-library modules, builtins, imported modules, or important domain names.

Recommended local checks:

```bash
ruff format .
ruff check --fix .
pyright
ty check
uv run pytest
```

## Package and Import Structure

### Package Layout

- Use a standard `src/` layout for importable packages when the project is packaged.
- Include `__init__.py` in normal package directories.
- Keep `__init__.py` lightweight: expose stable public APIs only; avoid heavy imports and side effects.
- Avoid namespace packages unless the project explicitly needs them.

```text
src/
└── mypackage/
    ├── __init__.py
    ├── core.py
    └── adapters/
        ├── __init__.py
        └── filesystem.py
```

## Typing Guidelines

### General Typing

- Annotate all public functions, methods, class attributes, and module-level constants.
- Prefer precise types over `Any`.
- Use `typing.Any` only at validated boundaries or when integrating with truly dynamic APIs.
- Prefer `object` over `Any` when the value is intentionally opaque.
- Use `TypeAlias` for complex reusable type expressions.
- Use `Protocol` for structural interfaces and dependency inversion.
- Use `Literal` for finite string modes and states.
- Use `TypedDict` or `pydantic` models for structured dictionaries crossing boundaries.
- For NumPy arrays, prefer `npt.NDArray[...]` to document dtype expectations.
  - Example: distinguish raw images (`np.uint8`) vs normalized tensors/arrays (`np.float32`).

```python
import numpy as np
import numpy.typing as npt

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias


PathLike: TypeAlias = str | bytes
Mode = Literal["train", "eval", "predict"]


class SupportsPredict(Protocol):
    def predict(self, x: Sequence[float]) -> float: ...

def image_processor(images: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    ...
```

### Modern Syntax

- Import abstract collection types from `collections.abc` when values are consumed generically.
- Prefer `Self` for fluent APIs when supported by the project Python version.

```python
from collections.abc import Iterable
from typing import Self


class Builder:
    def add(self, values: Iterable[str]) -> Self:
        return self
```

## Function Design

- Keep functions small, cohesive, and testable.
- Avoid mutable default arguments.
- Avoid boolean flags that create multiple modes; use separate functions or Literal modes when clearer.
- Raise specific exceptions with actionable messages.

```python
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    values = [] if items is None else list(items)
    values.append(item)
    return values
```

## Class and Data Model Design

### Dataclass vs Pydantic

- Use Pydantic models for external input/output boundaries: config files, API payloads, CLI input, serialized data.
- Use `@dataclass(slots=True)` for internal immutable or lightweight domain data.
- Avoid using Pydantic as a general-purpose internal data container unless validation/serialization is needed.
- Prefer immutable data (`frozen=True`) when mutation is not required.

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field


class TrainConfigInput(BaseModel):
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
```

### Interfaces

- Use abstract base classes only when shared implementation or nominal hierarchy is important.
- Keep constructors lightweight; avoid I/O or GPU allocation in `__init__` unless explicitly documented.

## Error Handling and Logging

### Exception Policy

- Catch specific exceptions before broad exceptions.
- Do not use `assert` for runtime validation in production code.

```python
from pathlib import Path
import json


def load_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON file: {path}"
        raise ValueError(msg) from e

    if not isinstance(data, dict):
        msg = f"Expected JSON object: {path}"
        raise TypeError(msg)
    return data
```

### Logging Policy

- Use a module-specific logger: `logging.getLogger(__name__)`.
- Log an exception once, at the outermost boundary that can add useful operational context.
- Inner layers should raise or chain exceptions, not call `logger.exception()`.
- Do not log secrets, credentials, tokens, raw personal data, or sensitive file contents.

```python
import logging

logger = logging.getLogger(__name__)


def run_workflow(config_path: Path) -> None:
    try:
        config = load_json(config_path)
        execute(config)
    except Exception:
        logger.exception("Workflow failed: config_path=%s", config_path)
        raise
```

## Filesystem, Paths, and Serialization

- Use `pathlib.Path` for filesystem paths.
- Use explicit encodings for text I/O.
- Use atomic writes for important output files when partial writes are harmful.
- Use safe YAML loading (`yaml.safe_load`) for YAML.

## Performance Guidelines

- Write clear code first; optimize after measuring.
- Avoid unnecessary copies of large arrays, tensors, images, and data frames.
- Prefer vectorized `NumPy`/`PyTorch` operations over Python loops for numeric workloads.
- Be explicit about device placement and dtype in PyTorch code.
- Avoid hidden synchronization points in GPU code when performance matters.
- Do not introduce caching unless invalidation and memory growth are understood.

## Testing Guidelines

- Use pytest.
- Keep tests deterministic and independent.
- Prefer fixtures for setup shared by multiple tests.
- Use specific exception assertions with `match=`.
- Avoid broad `pytest.raises(Exception)`.
- Use simple assertions so failures are easy to diagnose.
- Add regression tests for bug fixes.
- Mark resource-heavy tests with the appropriate marker: `slow`, `native`, or `gpu`.
- Do not use class `Test*` for test classes; pytest will discover functions at the module level.

```python
import pytest


def test_negative_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        calculate(-1)
```

## Documentation and Comments

- Use docstrings for public modules, classes, functions, and non-obvious behavior.
- Prefer Google-style docstrings if the project has no stronger convention.
- Explain why, not what, in comments.
- Keep examples small and executable where practical.
- Update documentation when commands, public APIs, configuration, or behavior changes.

## AI Agent Anti-Patterns

Avoid these common mistakes:

- Large unrelated refactors while solving a small task.
- Adding compatibility layers that are not required by the target Python version.
- Introducing `Any`, `# type: ignore`, or `# noqa` to silence errors without justification.
- Creating generic utility modules with vague names like helpers.py for unrelated functions.
- Adding repeated defensive checks inside trusted internal functions.
- Logging the same exception at multiple layers.
- Hiding I/O, network access, GPU allocation, or global state inside seemingly pure functions.
- Adding dependencies for trivial functionality available in the standard library.

## Quick Checklist Before Returning Code

- Is the diff minimal and related to the task?
- Are public functions typed?
- Are imports clean and at top level?
- Are errors specific and chained where appropriate?
- Are filesystem paths represented with `Path`?
- Are tests added or updated for behavior changes?
- Can `Ruff`, `Pyright`/`Ty`, and `pytest` reasonably pass?
