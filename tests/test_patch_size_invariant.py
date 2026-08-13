"""Patch-size invariants.

The mechanistic analysis hinges on each model's declared
``patch_size_px``. This test pins the expected values so an accidental edit
in a wrapper file is caught immediately.

If any of these values change deliberately, update both the wrapper and this
test in the same commit, and document the source of truth in the wrapper
docstring.
"""

from __future__ import annotations

import pytest

from ftw_eval.model_zoo import get_model


EXPECTED_PATCH_SIZE_PX: dict[str, int | None] = {
    "clay-v1": 8,
    "prithvi-eo-2.0-300m": 16,
    "anysat": 2, # = patch_size_m (20m) / GSD (10m); see anysat.py
    "terramind-v1-base": 16,
    "presto": None, # pixel-based; no spatial patch
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_PATCH_SIZE_PX.items()))
def test_patch_size_pinned(name: str, expected: int | None) -> None:
    fm = get_model(name, device="cpu")
    assert fm.patch_size_px == expected, (
        f"{name}: expected patch_size_px={expected}, got {fm.patch_size_px}. "
        "If you intended to change this, update the test AND verify the new "
        "value against the upstream model config."
    )
