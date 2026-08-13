"""Smoke tests that touch each wrapper's preprocess() with a synthetic chip.

These are CPU-only and skip automatically when the model's optional extra is not
installed. They exercise the conversion path from Chip -> tensor batch, which
is where most silent bugs (band ordering, normalization shape, dtype) hide.
"""

from __future__ import annotations

import importlib.util

import pytest

from ftw_eval.data.chip import Chip
from ftw_eval.model_zoo import get_model


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.fixture
def chip_single():
    return Chip.synthetic(size_px=64)


@pytest.fixture
def chip_timeseries():
    return Chip.synthetic(size_px=32, time_steps=12)


@pytest.mark.skipif(not _have("claymodel"), reason="clay extra not installed")
@pytest.mark.model_clay
def test_clay_preprocess(chip_single):
    fm = get_model("clay-v1", device="cpu")
    batch = fm.preprocess(chip_single)
    assert "pixels" in batch
    assert batch["pixels"].shape[-1] == fm.default_input_size_px
    assert batch["latlon"].shape == (1, 4)


@pytest.mark.skipif(not _have("terratorch"), reason="prithvi extra not installed")
@pytest.mark.model_prithvi
def test_prithvi_preprocess(chip_single):
    fm = get_model("prithvi-eo-2.0-300m", device="cpu")
    batch = fm.preprocess(chip_single)
    assert "pixel_values" in batch
    # [B, C, T, H, W]
    assert batch["pixel_values"].dim() == 5
    assert batch["pixel_values"].shape[1] == 6


@pytest.mark.skipif(not _have("presto"), reason="presto extra not installed")
@pytest.mark.model_presto
def test_presto_preprocess(chip_timeseries):
    fm = get_model("presto", device="cpu")
    batch = fm.preprocess(chip_timeseries)
    assert "x" in batch
    # [HW, T, C]
    assert batch["x"].dim() == 3


@pytest.mark.skipif(not _have("anysat"), reason="anysat extra not installed")
@pytest.mark.model_anysat
def test_anysat_preprocess(chip_single):
    fm = get_model("anysat", device="cpu")
    batch = fm.preprocess(chip_single)
    assert "s2" in batch


@pytest.mark.skipif(not _have("terratorch"), reason="terramind extra not installed")
@pytest.mark.model_terramind
def test_terramind_preprocess(chip_single):
    fm = get_model("terramind-v1-base", device="cpu")
    batch = fm.preprocess(chip_single)
    assert "S2L2A" in batch
