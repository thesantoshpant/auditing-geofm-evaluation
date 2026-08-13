"""Prithvi-EO-2.0 wrapper (NASA / IBM).

Reference:
    https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
    https://arxiv.org/abs/2412.02732
    Loaded via terratorch's BACKBONE_REGISTRY.

Prithvi-EO-2.0 expects 6 HLS-aligned bands (Blue, Green, Red, Narrow-NIR, SWIR1, SWIR2).
The HF config.json reuses HLS column names (B02..B07), which is **NOT** the same
as Sentinel-2 numbering. The correct S2 → HLS mapping for Narrow-NIR is B8A,
not B05 (B05 in S2 is rededge1). We resolve this explicitly.

Patch tokenization: 16 px on a 224 px input. ``patch_size=[1, 16, 16]`` per the
HF config. Single-time inference uses T=1.
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


# S2 band names that supply each Prithvi/HLS expected channel.
PRITHVI_S2_BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Canonical means/stds from HF config.json (raw HLS scale, i.e. reflectance * 10000).
# Source: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/config.json
PRITHVI_MEANS = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32
)
PRITHVI_STDS = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32
)


class PrithviFoundationModel(FoundationModel):
    name: ClassVar[str] = "prithvi-eo-2.0-300m"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 224
    patch_size_px: ClassVar[int | None] = 16
    pretrained_id: ClassVar[str | None] = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    backbone_name: ClassVar[str] = "prithvi_eo_v2_300"
    pooling_method: ClassVar[str] = "mean_no_cls"
    has_cls_token: ClassVar[bool] = True

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "Prithvi deps not installed. Run: pip install -e '.[prithvi]'"
            ) from e

        self._model = BACKBONE_REGISTRY.build(self.backbone_name, pretrained=True)
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        s2 = chip.select_s2_bands(PRITHVI_S2_BANDS).astype(np.float32)
        # Chip is in reflectance/10000; Prithvi stats are in raw HLS units.
        s2 = s2 * 10_000.0
        s2 = (s2 - PRITHVI_MEANS[:, None, None]) / PRITHVI_STDS[:, None, None]
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0) # [1, 6, H, W]
        t = F.interpolate(
            t,
            size=(self.default_input_size_px, self.default_input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        # Prithvi expects [B, C, T, H, W] with T=1 for single-time inputs.
        t = t.unsqueeze(2)
        return {"pixel_values": t}

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        x = batch["pixel_values"]
        with torch.no_grad():
            # terratorch's Prithvi backbone returns either a list of stage
            # outputs (if neck enabled) or a single tensor of patch tokens.
            # We probe both shapes.
            try:
                out = self._model.forward_features(x)
            except AttributeError:
                out = self._model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.dim() == 4:
            out = out.flatten(2).transpose(1, 2) # [B, N, D]
        # PrithviViT prepends a CLS token at index 0. Drop it from both the
        # pooled feature and the returned token grid so patch-grid
        # analysis sees a clean 14x14 = 196-token grid, and the pooled vector
        # doesn't average a non-spatial token into the spatial mean.
        patch_tokens = out[:, 1:]
        features = patch_tokens.mean(dim=1)
        return ModelOutput(
            tokens=patch_tokens if return_tokens else None,
            features=features,
            attention=None,
        )

    def encode_per_layer(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run the encoder once and return a list of per-block output tensors.

        PrithviViT (terratorch) exposes self._model.blocks as a ModuleList
        of 24 Block instances. Each Block has a proper forward, so we use
        register_forward_hook on each block individually. Returned tensors
        are [B, 1+L, D] (CLS at index 0).
        """
        if not self._loaded:
            self.load()
        blocks = self._model.blocks
        captured: list[torch.Tensor] = []
        handles = []

        def make_hook(_):
            def hook(_module, _inp, output):
                t = output[0] if isinstance(output, (tuple, list)) else output
                captured.append(t.detach().clone())
            return hook

        for blk in blocks:
            handles.append(blk.register_forward_hook(make_hook(None)))
        try:
            x = batch["pixel_values"]
            with torch.no_grad():
                try:
                    _ = self._model.forward_features(x)
                except AttributeError:
                    _ = self._model(x)
        finally:
            for h in handles:
                h.remove()
        return captured
