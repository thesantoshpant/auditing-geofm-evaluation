"""Model zoo registry.

Usage:
    from ftw_eval.model_zoo import get_model, list_models
    fm = get_model("clay-v1", device="cuda")
    out = fm.predict(chip)

Lazy-loading: importing this module does NOT import any model dependency.
The wrappers themselves import their deps inside ``load()``, so the package
remains importable even when only a subset of model extras are installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ftw_eval.model_zoo.base import (
    FoundationModel,
    FoundationModelNotInstalledError,
    ModelOutput,
)

if TYPE_CHECKING:
    pass


_REGISTRY: dict[str, tuple[str, str]] = {
    "clay-v1":              ("ftw_eval.model_zoo.clay",      "ClayFoundationModel"),
    "prithvi-eo-2.0-300m":  ("ftw_eval.model_zoo.prithvi",   "PrithviFoundationModel"),
    "presto":               ("ftw_eval.model_zoo.presto",    "PrestoFoundationModel"),
    "anysat":               ("ftw_eval.model_zoo.anysat",    "AnySatFoundationModel"),
    "anysat-dense":         ("ftw_eval.model_zoo.anysat",    "AnySatDenseFoundationModel"),
    "terramind-v1-base":    ("ftw_eval.model_zoo.terramind", "TerraMindFoundationModel"),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model(name: str, device: str = "cuda", **kwargs) -> FoundationModel:
    """Instantiate a wrapper by name. Does not call ``load()``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    module_path, class_name = _REGISTRY[name]
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls(device=device, **kwargs)


__all__ = [
    "FoundationModel",
    "FoundationModelNotInstalledError",
    "ModelOutput",
    "get_model",
    "list_models",
]
