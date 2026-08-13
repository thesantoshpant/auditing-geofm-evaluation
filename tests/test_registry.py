"""Registry-level smoke tests. CPU-only — does not load any model weights."""

from __future__ import annotations

import pytest

from ftw_eval.model_zoo import FoundationModel, get_model, list_models


EXPECTED = {
    "clay-v1", "prithvi-eo-2.0-300m", "presto", "anysat", "anysat-dense",
    "terramind-v1-base",
}


def test_list_models_returns_expected_set():
    assert set(list_models()) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_get_model_instantiates_without_loading(name):
    fm = get_model(name, device="cpu")
    assert isinstance(fm, FoundationModel)
    assert fm.name == name
    assert not fm.loaded
    info = fm.describe()
    assert info["name"] == name
    assert info["loaded"] is False


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("not-a-real-model")
