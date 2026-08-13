"""TerraMind wrapper (IBM / ESA, 2025).

Reference:
    https://ibm.github.io/terramind/
    HF: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base  (lowercase 'base')
    Loaded via terratorch's BACKBONE_REGISTRY.

TerraMind is an any-to-any generative FM across 9 modalities. For this project
we use its S2L2A encoder for representation extraction. The model is invoked
directly (``__call__``) with a dict of modality tensors; ``forward_features``
is not exposed on the terratorch backbone.

Patch tokenization: 16 px on a 224 px input (ViT-B-style 14×14 = 196 tokens).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from ftw_eval.data.chip import Chip
from ftw_eval.model_zoo.base import (
    FoundationModel,
    FoundationModelNotInstalledError,
    ModelOutput,
)


# TerraMind S2L2A patch-embed expects 12 channels per token. The proj weight
# shape is 3072 x 768 (3072 = 16*16*12). Supplying 10 bands fails with a matmul
# shape mismatch. Canonical ESA L2A order (excludes B10 which is L1C-only).
TERRAMIND_S2_BANDS: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]

# TerraMind's terratorch ViT *encoder* (terramind_v1_base) does NOT standardize
# its input — terratorch normally applies these per-band stats in the
# datamodule, which is bypassed when we build the bare backbone via
# BACKBONE_REGISTRY.build. We must therefore standardize here, exactly as Clay
# and Prithvi do. Stats are TerraMind v1 `untok_sen2l2a@224` pretraining
# mean/std (DN units = reflectance*10000), band order identical to
# TERRAMIND_S2_BANDS above (COASTAL..SWIR2). Verified from
# terratorch/models/backbones/terramind/model/terramind_register.py.
# Omitting this (the original bug) fed reflectance/10000 ~[0,0.5] to a model
# expecting standardized N(0,1) inputs -> off-distribution features.
TERRAMIND_S2L2A_MEAN: list[float] = [
    1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
    2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648,
]
TERRAMIND_S2L2A_STD: list[float] = [
    2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311,
]


class TerraMindFoundationModel(FoundationModel):
    name: ClassVar[str] = "terramind-v1-base"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 224
    patch_size_px: ClassVar[int | None] = 16
    pretrained_id: ClassVar[str | None] = "ibm-esa-geospatial/TerraMind-1.0-base"
    backbone_name: ClassVar[str] = "terramind_v1_base"
    pooling_method: ClassVar[str] = "mean"
    has_cls_token: ClassVar[bool] = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "TerraMind deps not installed. Run: pip install -e '.[terramind]'"
            ) from e

        self._model = BACKBONE_REGISTRY.build(
            self.backbone_name,
            pretrained=True,
            modalities=["S2L2A"],
        )
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        # chip stores reflectance/10000; TerraMind stats are in DN units, so
        # rescale by 1e4 then standardize (x - mean)/std per band.
        s2 = chip.select_s2_bands(TERRAMIND_S2_BANDS).astype(np.float32) * 10000.0
        mean = np.asarray(TERRAMIND_S2L2A_MEAN, dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(TERRAMIND_S2L2A_STD, dtype=np.float32).reshape(-1, 1, 1)
        s2 = (s2 - mean) / std
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        t = F.interpolate(
            t,
            size=(self.default_input_size_px, self.default_input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        return {"S2L2A": t}

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        with torch.no_grad():
            out = self._model(batch)
        # terratorch TerraMindViT.forward returns list[Tensor], one tensor per
        # encoder block. The last element is the LayerNorm'd final output.
        # See terratorch/models/backbones/terramind/model/terramind_vit.py.
        tokens = out[-1] if isinstance(out, (list, tuple)) else out
        if tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, D]
        features = tokens.mean(dim=1)
        return ModelOutput(
            tokens=tokens if return_tokens else None,
            features=features,
            attention=None,
        )

    def encode_per_layer(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Return per-block outputs for the 12 encoder blocks.

        TerraMindViT.forward already returns a list[Tensor] of length 12,
        one tensor per block (the last is LayerNorm'd). We pass them
        through unchanged so each layer can be probed independently.
        """
        if not self._loaded:
            self.load()
        with torch.no_grad():
            out = self._model(batch)
        if isinstance(out, (list, tuple)):
            layers = list(out)
        else:
            layers = [out]
        # Each tensor is [B, N, D] for transformer outputs or [B, D, H, W]
        # if the backbone returns spatial maps. Normalize to [B, N, D].
        norm = []
        for t in layers:
            if t.dim() == 4:
                t = t.flatten(2).transpose(1, 2)
            norm.append(t.detach().clone())
        return norm
