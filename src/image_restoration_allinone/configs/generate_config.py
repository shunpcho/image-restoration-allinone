import inspect
from dataclasses import make_dataclass
from typing import Any, get_type_hints


def dataclass_from_class(cls: type) -> type:
    """Create a dataclass from a class with type hints."""
    hints = get_type_hints(cls.__init__)
    sig = inspect.signature(cls.__init__)
    fields: list[Any] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        annotation = hints.get(name, object)
        default = param.default if param.default != inspect._empty else ...  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]

        fields.append((name, annotation, default))

    def from_optional_kwargs(cls: type, **kwargs: dict[str, Any]) -> type:
        return cls(**{key: value for key, value in kwargs.items() if key in {name for name, _, _ in fields}})

    return make_dataclass(
        cls.__name__ + "Config",
        fields,
        namespace={"from_optional_kwargs": classmethod(from_optional_kwargs)},
        frozen=True,
        slots=True,
    )
