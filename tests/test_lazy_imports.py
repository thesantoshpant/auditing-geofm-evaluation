"""Verify the model_zoo lazy-import contract.

Importing ``ftw_eval.model_zoo`` and instantiating a wrapper must NOT
pull in the FM-specific libraries (claymodel, terratorch, presto). Those
are only imported inside ``load()`` so that:
  - the registry stays importable on the server even mid-install
  - tests that don't need a model never pay the import cost
"""

from __future__ import annotations

import importlib
import sys


_FM_LIBS = {"claymodel", "terratorch", "presto"}


def test_importing_registry_does_not_load_fm_libs():
    for mod in list(sys.modules):
        if mod in _FM_LIBS or mod.split(".")[0] in _FM_LIBS:
            del sys.modules[mod]

    importlib.import_module("ftw_eval.model_zoo")
    leaked = {m for m in sys.modules if m.split(".")[0] in _FM_LIBS}
    assert not leaked, f"Lazy-import contract broken: {leaked} imported eagerly"


def test_instantiating_wrapper_does_not_load_fm_libs():
    from ftw_eval.model_zoo import get_model

    for mod in list(sys.modules):
        if mod.split(".")[0] in _FM_LIBS:
            del sys.modules[mod]

    fm = get_model("clay-v1", device="cpu")
    assert not fm.loaded
    leaked = {m for m in sys.modules if m.split(".")[0] in _FM_LIBS}
    assert not leaked, f"get_model triggered FM lib import: {leaked}"
