"""Tests for the Chip dataclass. CPU-only, no FM deps required."""

from __future__ import annotations

import numpy as np
import pytest

from ftw_eval.data.chip import S2_BAND_ORDER, Chip


def test_synthetic_chip_shape():
    chip = Chip.synthetic(size_px=64)
    assert chip.s2.shape == (12, 64, 64)  # canonical L2A band count
    assert chip.s1.shape == (2, 64, 64)
    assert chip.dem.shape == (64, 64)
    assert chip.height_px == 64
    assert chip.width_px == 64
    assert chip.height_m == 640.0
    chip.validate()


def test_synthetic_timeseries():
    chip = Chip.synthetic(size_px=32, time_steps=12)
    assert chip.has_time
    assert chip.s2.shape == (12, 12, 32, 32)  # [T=12, C=12, H, W]


def test_select_s2_bands():
    chip = Chip.synthetic(size_px=16)
    rgb = chip.select_s2_bands(["B04", "B03", "B02"])
    assert rgb.shape == (3, 16, 16)
    np.testing.assert_array_equal(rgb[0], chip.s2[S2_BAND_ORDER.index("B04")])


def test_select_missing_band_raises():
    chip = Chip.synthetic(size_px=16)
    with pytest.raises(ValueError, match="missing"):
        chip.select_s2_bands(["B04", "B99"])


def test_validate_rejects_bad_lat():
    chip = Chip.synthetic()
    chip.lat = 1000.0
    with pytest.raises(ValueError, match="lat"):
        chip.validate()


def test_validate_rejects_band_count_mismatch():
    chip = Chip.synthetic(size_px=16)
    chip.s2_bands = ["B02", "B03"]  # claim 2 bands but array has 12
    with pytest.raises(ValueError, match="s2 has"):
        chip.validate()
