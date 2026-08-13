"""AnySat wrapper (CVPR 2025, Astruc et al.).

Reference (source of truth):
    https://raw.githubusercontent.com/gastruc/AnySat/main/hubconf.py
    https://raw.githubusercontent.com/gastruc/AnySat/main/README.md
    Loaded via ``torch.hub.load('gastruc/anysat', 'anysat', pretrained=True)``.

AnySat enforces 5D inputs per modality: ``[B, T, C, H, W]``. For each
time-series modality it also requires a ``{modality}_dates`` tensor of shape
``[B, T]`` containing day-of-year integers (01/01 = 0, 31/12 = 364).

``patch_size`` is specified at call time in meters and must be a multiple of 10.
README examples use ``patch_size=10`` and ``patch_size=20``. ``40`` is NOT
documented as a supported value. We default to **20 m** so the token tile is
2x2 pixels at S2 10 m GSD.

## Output mode caveat ( sign-off, revisit)

AnySat's ``output='patch'`` mode in the released hub checkpoint emits a 3D
tensor of shape ``[B, P, D]`` with P = 92160 = 96 * 96 * 10 for a 240-px input
at patch_size=20. The 10 factor is the S2 band count, i.e. AnySat returns per-
band sub-patch tokens, NOT a single ViT-style 14x14 patch grid. This means:

1. Memory at ``output='patch'`` is ~27 GB on a single 240-px chip (A6000 fits
   single-chip but not batched).
2. The token grid is not directly comparable to Clay/Prithvi/TerraMind's
   single-modality ViT tokens, breaking the patch-boundary analysis
   premise for AnySat specifically.

For sign-off we default to ``output='tile'`` which returns a single
pooled vector per chip ``[B, D]`` (cheap, fast, suitable for zero-
shot eval). The mechanistic analysis will need a follow-up audit:
either AnySat is included via ``output='dense'`` (and we explain the per-band
token structure carefully), or AnySat is documented as a "tile-only baseline"
that gives a competitive zero-shot reference but cannot be patch-boundary
probed. Decide this after chips are in hand.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


class AnySatFoundationModel(FoundationModel):
    name: ClassVar[str] = "anysat"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 240
    # patch_size_px = patch_size_m / gsd_m = 20 / 10 = 2 at S2 native.
    patch_size_px: ClassVar[int | None] = 2
    pretrained_id: ClassVar[str | None] = "gastruc/anysat"
    patch_size_m: ClassVar[int] = 20 # 20 m is the largest documented value in the README
    # : 'tile' returns [B, D]; cheap, comparable to other FMs' pooled feature.
    # may revisit and switch to 'dense' or 'all' once we audit token shape.
    output_mode: ClassVar[str] = "tile"
    pooling_method: ClassVar[str] = "tile"
    has_cls_token: ClassVar[bool] = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            self._model = torch.hub.load(
                "gastruc/anysat",
                "anysat",
                pretrained=True,
                flash_attn=False,
                trust_repo=True,
            )
        except Exception as e:
            raise FoundationModelNotInstalledError(
                f"Failed to load AnySat via torch.hub: {e}. Confirm internet access "
                "and that the gastruc/anysat repo is reachable from the server."
            ) from e
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    # AnySat's S2 projector was trained on 10 bands (the 10/20 m bands; B01 and
    # B09 are 60 m bands that AnySat does not consume). Selecting the canonical
    # 10-band subset prevents a matmul shape mismatch against the [10, 96] weight.
    S2_BANDS: ClassVar[list[str]] = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    ]

    # AnySat REQUIRES standardized input: (data - mean)/std (README line 74;
    # demo.ipynb). Stats are GeoPlex S2 mean/std in DN units (reflectance*10000),
    # band order matching S2_BANDS above (B02..B12, 60 m B01/B09 excluded).
    # Omitting this (the original bug) fed raw reflectance/10000 ~[0,0.5] to a
    # model expecting ~N(0,1) -> off-distribution features.
    S2_MEAN: ClassVar[list[float]] = [
        4304.32958984375, 4159.2666015625, 4057.776611328125, 4328.951171875,
        4571.22119140625, 4644.87109375, 4837.2470703125, 4700.2578125,
        2823.264404296875, 2319.97021484375,
    ]
    S2_STD: ClassVar[list[float]] = [
        3537.99755859375, 3324.23486328125, 3270.070068359375, 3250.530029296875,
        2897.391357421875, 2754.4970703125, 2821.521484375, 2625.952392578125,
        1731.56298828125, 1549.3028564453125,
    ]

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        if chip.s2 is None:
            raise ValueError("AnySat preprocessing requires S2 data.")
        # chip stores reflectance/10000; AnySat stats are in DN units, so
        # rescale by 1e4 then standardize per band.
        s2 = chip.select_s2_bands(self.S2_BANDS).astype(np.float32) * 10000.0
        _mean = np.asarray(self.S2_MEAN, dtype=np.float32).reshape(-1, 1, 1)
        _std = np.asarray(self.S2_STD, dtype=np.float32).reshape(-1, 1, 1)
        s2 = (s2 - _mean) / _std
        t = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        if t.dim() == 4:
            t = F.interpolate(
                t,
                size=(self.default_input_size_px, self.default_input_size_px),
                mode="bilinear",
                align_corners=False,
            )
        # AnySat enforces 5D inputs per modality: [B, T, C, H, W]. T=1 for single-time.
        t = t.unsqueeze(1)
        # s2_dates: day-of-year integers, shape [B, T] (per README convention).
        date = chip.date or datetime(2024, 6, 15, tzinfo=timezone.utc)
        doy = date.timetuple().tm_yday - 1 # 0..364
        s2_dates = torch.tensor([[doy]], dtype=torch.long, device=self.device)
        return {"s2": t, "s2_dates": s2_dates}

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        with torch.no_grad():
            out = self._model(
                batch,
                patch_size=self.patch_size_m,
                output=self.output_mode,
            )
        # In 'tile' mode AnySat returns a single [B, D] pooled vector per chip.
        # No token grid is exposed. mechanistic analysis on AnySat
        # requires switching this wrapper to 'dense' or 'all'; until then,
        # token-level analysis on AnySat is disabled by design.
        if out.dim() == 2:
            features = out
            tokens = None
        elif out.dim() == 3:
            tokens = out if return_tokens else None
            features = out.mean(dim=1)
        elif out.dim() == 4:
            # [B, D, H, W] -> [B, N, D]
            out = out.flatten(2).transpose(1, 2)
            tokens = out if return_tokens else None
            features = out.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unexpected AnySat output rank {out.dim()} with shape {tuple(out.shape)}"
            )
        return ModelOutput(tokens=tokens, features=features, attention=None)


class AnySatDenseFoundationModel(AnySatFoundationModel):
    """AnySat in dense (per-pixel) mode.

    ``output='dense'`` returns a [B, H, W, D] feature map at the FM's input
    resolution (240x240 at patch_size=20, D=1536) — i.e. a per-pixel dense
    embedding, NOT a coarse patch grid. This makes AnySat usable in the
    per-pixel methodology: the dense map is already pixel-aligned (one
    feature vector per input pixel), so ``patch_size_px = 1``.

    Requires ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` to fit the
    transient ~9 GB activation on a contended shared GPU.

    We flatten the [B, H, W, D] map to [B, H*W, D] so the downstream
    per-pixel extractor's ``reshape_to_grid`` (which reshapes [N, D] to a
    square [sqrt(N), sqrt(N), D] grid) reconstructs the 240x240 grid
    exactly (240*240 = 57600 is a perfect square).
    """

    name: ClassVar[str] = "anysat-dense"
    # Dense map is at input resolution: one token per input pixel.
    patch_size_px: ClassVar[int | None] = 1
    output_mode: ClassVar[str] = "dense"
    pooling_method: ClassVar[str] = "dense"

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        with torch.no_grad():
            out = self._model(batch, patch_size=self.patch_size_m, output="dense")
        # dense output is [B, H, W, D]. Flatten spatial dims to [B, H*W, D].
        if out.dim() == 4:
            b, h, w, d = out.shape
            tokens = out.reshape(b, h * w, d)
        elif out.dim() == 3:
            tokens = out
        else:
            raise RuntimeError(
                f"Unexpected AnySat dense output rank {out.dim()} shape {tuple(out.shape)}"
            )
        features = tokens.mean(dim=1)
        return ModelOutput(
            tokens=tokens if return_tokens else None,
            features=features,
            attention=None,
        )
