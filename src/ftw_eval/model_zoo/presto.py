"""Presto wrapper (NASA Harvest).

Reference:
    https://github.com/nasaharvest/presto
    https://arxiv.org/abs/2304.14065

Presto is a **pixel-time-series** model, not an image patch model. The natural
unit is a single pixel's monthly time series across S1+S2+ERA5+SRTM. Encoder
signature:

    encoder.forward(
        x: [B, T, len(NORMED_BANDS)],
        dynamic_world: [B, T], # required positional
        latlons: [B, 2],
        mask: [B, T, len(NORMED_BANDS)] | None,
        month: int,
        eval_task: bool = True,
    )

Presto's band layout is dictated by ``presto.dataops.BANDS`` /
``NORMED_BANDS`` / ``BANDS_GROUPS_IDX``, which concatenates S1, S2, ERA5, and
SRTM channels per timestep. A correct adapter requires:

  1. Building the canonical 17-channel layout (or whatever the installed
     version exposes), with masking for missing modalities.
  2. Supplying ``dynamic_world`` as a per-timestep integer label tensor
     (``DYNAMIC_WORLD_NULL_CLASS`` = unknown) of shape ``[B, T]``.
  3. Choosing the correct ``month`` index per acquisition.

That logic is non-trivial and depends on the installed Presto version. Presto is
not part of the evaluated model set in this artifact and is skipped by the
installation verifier by default. The wrapper below records the model contract,
but ``preprocess()`` and ``encode()`` intentionally raise ``NotImplementedError``
until a version-pinned pixel-time-series adapter is added.

``patch_size_px`` is ``None`` because Presto is pixel based.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from ftw_eval.data.chip import Chip
from ftw_eval.model_zoo.base import (
    FoundationModel,
    FoundationModelNotInstalledError,
    ModelOutput,
)


class PrestoFoundationModel(FoundationModel):
    name: ClassVar[str] = "presto"
    required_modalities: ClassVar[set[str]] = {"s2"}
    default_input_size_px: ClassVar[int] = 1  # per-pixel
    patch_size_px: ClassVar[int | None] = None
    pretrained_id: ClassVar[str | None] = "nasaharvest/presto"
    pooling_method: ClassVar[str] = "pixel"
    has_cls_token: ClassVar[bool] = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from presto import Presto
        except ImportError as e:
            raise FoundationModelNotInstalledError(
                "Presto is not part of the shared environment. See ARTIFACT.md "
                "for the documented dependency conflict and separate-environment requirement."
            ) from e

        self._model = Presto.load_pretrained()
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def preprocess(self, chip: Chip) -> dict[str, torch.Tensor]:
        # A correct chip-to-Presto adapter must:
        # - map Chip bands to NORMED_BANDS order
        # - construct dynamic_world ([B, T], int, DYNAMIC_WORLD_NULL_CLASS)
        # - select month index from chip.date
        # See presto/dataops.py for the exact contract.
        raise NotImplementedError(
            "Presto is not evaluated in this artifact because it requires a "
            "version-pinned pixel-time-series adapter. See ARTIFACT.md."
        )

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        return_tokens: bool = True,
        return_attention: bool = False,
    ) -> ModelOutput:
        if not self._loaded:
            self.load()
        # The Presto encoder call will be:
        # encoder(x, dynamic_world, latlons=latlons, mask=mask, month=month)
        # after preprocess() supplies the required dataops tensors.
        raise NotImplementedError(
            "Presto encode() requires the same pixel-time-series adapter as "
            "preprocess(); Presto is skipped in this artifact."
        )
