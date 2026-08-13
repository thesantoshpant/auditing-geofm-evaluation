"""Abstract foundation model interface.

Every wrapped FM implements this protocol so downstream code (zero-shot eval,
LoRA adaptation, TokenPool method, mechanistic probing) treats them uniformly.

Notes on patch tokenization
---------------------------
``patch_size_px`` is the **most important** piece of metadata for this project.
The mechanistic analysis uses it to compute per-field patch-boundary
overlap. Verify this value against each FM's source carefully — a wrong value
silently breaks the central claim of the paper.

For pixel-based models (Presto) where there is no spatial patch, return
``None`` and document the alternative tokenization explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from ftw_eval.data.chip import Chip


@dataclass
class ModelOutput:
    """Standardized output container.

    Attributes:
        tokens: Per-patch embeddings [B, N, D] when the model exposes them.
        features: Pooled embedding [B, D] suitable as input to a probe head.
        attention: Optional last-layer attention [B, heads, N, N] for mechanistic analysis.
        extras: Anything model-specific.
    """

    tokens: torch.Tensor | None
    features: torch.Tensor
    attention: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] | None = None


class FoundationModel(ABC):
    """Common interface across geospatial FMs."""

    name: ClassVar[str]
    required_modalities: ClassVar[set[str]]
    default_input_size_px: ClassVar[int] = 224
    patch_size_px: ClassVar[int | None] = None
    pretrained_id: ClassVar[str | None] = None
    # Telemetry: how the wrapper builds the pooled `features` from tokens.
    # Values: 'cls', 'mean', 'mean_no_cls', 'tile', 'pixel'.
    pooling_method: ClassVar[str] = "mean"
    has_cls_token: ClassVar[bool] = False

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights onto ``self.device``. Idempotent."""

    @abstractmethod
    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        """Convert a ``Chip`` into the model's expected input dict."""

    @abstractmethod
    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        """Run the model in inference mode and return a ``ModelOutput``."""

    @torch.no_grad()
    def predict(
        self,
        chip: Chip,
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        batch = self.preprocess(chip)
        return self.encode(batch, return_tokens=return_tokens, return_attention=return_attention)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pretrained_id": self.pretrained_id,
            "required_modalities": sorted(self.required_modalities),
            "default_input_size_px": self.default_input_size_px,
            "patch_size_px": self.patch_size_px,
            "pooling_method": self.pooling_method,
            "has_cls_token": self.has_cls_token,
            "device": self.device,
            "loaded": self._loaded,
        }


class FoundationModelNotInstalledError(RuntimeError):
    """Raised when a wrapper is instantiated but its optional deps are missing."""
