"""Clay v1.5 wrapper (Radiant Earth).

Reference (canonical, source of truth):
    https://raw.githubusercontent.com/Clay-foundation/model/main/claymodel/model.py
    https://clay-foundation.github.io/model/tutorials/wall-to-wall.html
    Canonical metadata: configs/metadata.yaml in the Clay repo.

NOTE: the older basic_use.html doc shows `encoder(chips, timestamps, wavelengths)`
as three positional args - that contradicts the actual source. The Encoder.forward
in claymodel/model.py takes a single ``datacube`` dict. Trust the source.

Datacube keys expected by Encoder.forward:
    pixels   : [B, C, H, W]  normalized reflectance, C=10
    time     : [B, 4]        [sin(week·2π/52), cos(...), sin(hour·2π/24), cos(...)]
    latlon   : [B, 4]        [sin(lat_rad), cos(...), sin(lon_rad), cos(...)]
    gsd      : 0-d tensor    meters per pixel
    waves    : [C]           Sentinel-2 central wavelengths in nm
    platform : str           e.g. "sentinel-2-l2a" (not strictly required by encoder)

Encoder.forward returns a 4-tuple:
    (unmsk_patch [B, 1+L, D], unmsk_idx, msk_idx, msk_matrix)
with the CLS token at index 0 of dim 1 and patch tokens at 1:.

At inference we set ``encoder.mask_ratio = 0`` so all patches are returned.

Patch tokenization: 8 px on a 256 px input.
"""

from __future__ import annotations

import contextlib
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
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


CLAY_METADATA_URL = (
    "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"
)
CLAY_CACHE_DIR = Path.home() / ".cache" / "ftw_eval" / "clay"


# Canonical from Clay-foundation/model:configs/metadata.yaml (raw L2A scale).
# Band order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
CLAY_S2_MEANS = np.array(
    [1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0],
    dtype=np.float32,
)
CLAY_S2_STDS = np.array(
    [1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0],
    dtype=np.float32,
)

# Sentinel-2 central wavelengths (nm) for B02..B12 in Clay's order.
CLAY_S2_WAVELENGTHS_NM = np.array(
    [493.0, 560.0, 665.0, 704.0, 740.0, 783.0, 833.0, 865.0, 1610.0, 2190.0],
    dtype=np.float32,
)


def _ensure_clay_metadata_cwd() -> Path:
    """Cache Clay's configs/metadata.yaml and return the dir to chdir into.

    The Clay v1.5 checkpoint stores ``metadata_path='configs/metadata.yaml'``
    as a relative path in its saved hparams, so load_from_checkpoint fails to
    resolve the file when CWD is anything other than the Clay source tree.
    We cache the YAML under ~/.cache/ftw_eval/clay/configs/ and chdir to
    its parent during load.
    """
    configs_dir = CLAY_CACHE_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = configs_dir / "metadata.yaml"
    if not metadata_file.exists():
        urllib.request.urlretrieve(CLAY_METADATA_URL, metadata_file)
    return CLAY_CACHE_DIR


@contextlib.contextmanager
def _chdir(path: Path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


class ClayFoundationModel(FoundationModel):
    name: ClassVar[str] = "clay-v1"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 256
    patch_size_px: ClassVar[int | None] = 8
    pretrained_id: ClassVar[str | None] = "made-with-clay/Clay"
    checkpoint_filename: ClassVar[str] = "v1.5/clay-v1.5.ckpt"
    pooling_method: ClassVar[str] = "cls"
    has_cls_token: ClassVar[bool] = True

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from claymodel.module import ClayMAEModule
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "Clay deps not installed. Run: pip install -e '.[clay]'"
            ) from e

        ckpt_path = hf_hub_download(
            repo_id=self.pretrained_id, filename=self.checkpoint_filename
        )
        cache_dir = _ensure_clay_metadata_cwd()
        with _chdir(cache_dir):
            self._module = ClayMAEModule.load_from_checkpoint(ckpt_path)
        self._module = self._module.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        chip.validate()
        wanted = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        s2 = chip.select_s2_bands(wanted).astype(np.float32)
        # Chip stores values divided by 10000; Clay's normalization is in raw L2A.
        s2 = s2 * 10_000.0
        s2 = (s2 - CLAY_S2_MEANS[:, None, None]) / CLAY_S2_STDS[:, None, None]
        pixels = torch.from_numpy(s2).to(self.device).to(self.dtype).unsqueeze(0)
        pixels = F.interpolate(
            pixels,
            size=(self.default_input_size_px, self.default_input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        date = chip.date or datetime(2024, 6, 15, tzinfo=timezone.utc)
        week = date.isocalendar().week * 2 * np.pi / 52.0
        hour = date.hour * 2 * np.pi / 24.0
        lat_r = np.radians(chip.lat)
        lon_r = np.radians(chip.lon)
        time_t = torch.tensor(
            [[np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)]],
            dtype=self.dtype, device=self.device,
        )
        latlon_t = torch.tensor(
            [[np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)]],
            dtype=self.dtype, device=self.device,
        )
        waves = torch.from_numpy(CLAY_S2_WAVELENGTHS_NM).to(self.device).to(self.dtype)
        gsd = torch.tensor(chip.gsd_m, dtype=self.dtype, device=self.device)  # 0-d
        return {
            "pixels": pixels,
            "time": time_t,
            "latlon": latlon_t,
            "gsd": gsd,
            "waves": waves,
            "platform": "sentinel-2-l2a",
        }

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        # Disable masking for inference so we get all patch tokens back.
        encoder = self._module.model.encoder
        encoder.mask_ratio = 0.0
        with torch.no_grad():
            out = encoder(batch)
        # encoder returns (unmsk_patch, unmsk_idx, msk_idx, msk_matrix).
        unmsk_patch = out[0] if isinstance(out, (tuple, list)) else out
        cls_tok = unmsk_patch[:, 0]
        patch_tok = unmsk_patch[:, 1:]
        return ModelOutput(
            tokens=patch_tok if return_tokens else None,
            features=cls_tok,
            attention=None,
        )

    def encode_per_layer(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run the encoder once and return a list of (B, 1+L, D) tensors,
        one per transformer block.

        Implementation note: Clay v1.5 uses vit_pytorch's SimpleViT
        Transformer where ``transformer.layers`` is a ModuleList of
        [Attention, FeedForward] pairs. ModuleList has no .forward, so
        nn.Module.register_forward_hook does not fire. We monkey-patch
        ``transformer.forward`` for the duration of one call, which
        replicates vit_pytorch's loop and captures the post-block
        activation at each depth.

        Tokens are NOT masked at inference (mask_ratio=0 by
        construction).
        """
        if not self._loaded:
            self.load()
        encoder = self._module.model.encoder
        encoder.mask_ratio = 0.0
        transformer = encoder.transformer
        if not hasattr(transformer, "layers"):
            raise RuntimeError("Clay encoder.transformer has no .layers attribute")
        captured: list[torch.Tensor] = []
        original_forward = transformer.forward

        def patched_forward(x):
            for attn, ff in transformer.layers:
                x = attn(x) + x
                x = ff(x) + x
                captured.append(x.detach().clone())
            if hasattr(transformer, "norm"):
                return transformer.norm(x)
            return x

        transformer.forward = patched_forward
        try:
            with torch.no_grad():
                _ = encoder(batch)
        finally:
            transformer.forward = original_forward
        return captured
