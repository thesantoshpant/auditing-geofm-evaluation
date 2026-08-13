#!/usr/bin/env python
"""Per-pixel non-FM baseline. The apples-to-apples spectral-only comparison
to the per-pixel FM eval.

For each sample listed in features_per_pixel_meta.parquet, recompute a 30-d
spectral feature vector from a small (default 3x3) window centered on the
sampled pixel in the chip's S2 GeoTIFF, then train a Random Forest. This
keeps the sampling scheme identical to the per-pixel FM protocol, so the
RF and FMs see exactly the same set of pixels and the comparison is clean.

Usage:
    python scripts/eval_nonfm_baseline_per_pixel.py \
        --manifest data/chips/manifest_canonical.jsonl \
        --features-dir data/features_per_pixel_canonical \
        --out data/results/eval_per_pixel_canonical_nonfm_rf.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ftw_eval.data.labels import FieldSizeBin, read_manifest  # noqa: E402
from ftw_eval.utils.logging import banner, get_logger  # noqa: E402

log = get_logger("nonfm-pp")

IDX_BLUE = 1   # B02
IDX_RED = 3    # B04
IDX_NIR = 7    # B08


def features_for_pixel_window(arr: np.ndarray, r: int, c: int, half: int) -> np.ndarray:
    """30-d spectral feature vector for a (2*half+1)x(2*half+1) chunk."""
    h, w = arr.shape[1], arr.shape[2]
    r0 = max(0, r - half)
    r1 = min(h - 1, r + half)
    c0 = max(0, c - half)
    c1 = min(w - 1, c + half)
    chunk = arr[:, r0:r1 + 1, c0:c1 + 1]
    if chunk.size == 0:
        return np.zeros(30, dtype=np.float32)
    means = chunk.reshape(chunk.shape[0], -1).mean(axis=1)
    sds = chunk.reshape(chunk.shape[0], -1).std(axis=1)
    red = chunk[IDX_RED]
    nir = chunk[IDX_NIR]
    blue = chunk[IDX_BLUE]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    feats = np.concatenate([
        means, sds,
        np.array([float(ndvi.mean()), float(ndvi.std()),
                  float(ndvi.min()), float(ndvi.max())], dtype=np.float32),
        np.array([float(evi.mean()), float(evi.std())], dtype=np.float32),
    ])
    return feats.astype(np.float32)


def build_stratum(row) -> str:
    if row["label"] == 0:
        return f"neg_{row['district']}"
    return f"pos_{row['district']}_{row['size_bin']}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True,
                   help="dir with features_per_pixel_meta.parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clf", choices=["rf", "lr", "xgb"], default="rf")
    p.add_argument("--half-px", type=int, default=1,
                   help="half-size of the per-pixel context window (default 1 -> 3x3)")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--split", choices=["chip", "pixel"], default="chip",
                   help="chip-grouped (default, no leakage) or legacy pixel split")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.clf == "xgb":
        # Fail fast BEFORE the expensive raster feature-extraction loop if the
        # optional dep is missing, rather than crashing after it.
        try:
            import xgboost  # noqa: F401
        except ImportError:
            log.error("--clf xgb requires xgboost. Install into the venv: "
                      "./.venv/bin/pip install xgboost")
            return 2
    manifest = read_manifest(args.manifest)
    chip_lookup = {e.chip_id: e for e in manifest}
    meta = pd.read_parquet(args.features_dir / "features_per_pixel_meta.parquet")
    banner(f"Per-pixel non-FM baseline on {len(meta)} samples "
           f"(window={args.half_px*2+1}x{args.half_px*2+1})")

    by_chip: dict[str, list[int]] = defaultdict(list)
    for i, row in meta.iterrows():
        by_chip[row["chip_id"]].append(int(i))

    X = np.zeros((len(meta), 30), dtype=np.float32)
    for ci, (chip_id, idxs) in enumerate(by_chip.items(), 1):
        entry = chip_lookup.get(chip_id)
        if entry is None:
            continue
        with rasterio.open(entry.s2_path) as src:
            arr = src.read().astype(np.float32) / 10_000.0
        for idx in idxs:
            r = int(meta.iloc[idx]["pixel_r"])
            c = int(meta.iloc[idx]["pixel_c"])
            X[idx] = features_for_pixel_window(arr, r, c, args.half_px)
        if ci % 50 == 0:
            log.info("  processed %d/%d chips", ci, len(by_chip))

    y = meta["label"].to_numpy()
    bins = meta["size_bin"].to_numpy()
    chips = meta["chip_id"].to_numpy()
    strata = meta.apply(build_stratum, axis=1)
    counts = strata.value_counts()
    keep = strata.isin(counts[counts >= 2].index).to_numpy()
    # SPLIT-THEN-MASK parity with the FM eval (scripts/eval_per_pixel.py):
    # split on the SAME strata-kept set FIRST so the chip partition is
    # byte-identical to the FM eval, THEN drop all-zero (failed-load) rows
    # within each split. The previous mask-then-split order could move chips
    # between partitions and give the RF a different test set. 
    X, y, bins, chips = X[keep], y[keep], bins[keep], chips[keep]
    strata_v = strata[keep].to_numpy()
    valid = ~np.all(X == 0, axis=1)
    n_train_chips = n_test_chips = chip_overlap = None

    if args.split == "chip":
        # CHIP-GROUPED split (default): no spatial leakage. Required so the
        # spectral RF cannot memorize test pixels from same-chip train pixels.
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_pos, te_pos = next(gss.split(X, y, groups=chips))
        tr_pos = tr_pos[valid[tr_pos]]
        te_pos = te_pos[valid[te_pos]]
        Xtr, Xte = X[tr_pos], X[te_pos]
        ytr, yte = y[tr_pos], y[te_pos]
        bins_tr, bins_te = bins[tr_pos], bins[te_pos]
        n_train_chips = len(set(chips[tr_pos]))
        n_test_chips = len(set(chips[te_pos]))
        chip_overlap = len(set(chips[tr_pos]) & set(chips[te_pos]))
        log.info("  CHIP-GROUPED split: train chips=%d test chips=%d overlap=%d dropped_allzero=%d",
                 n_train_chips, n_test_chips, chip_overlap, int((~valid).sum()))
    else:
        idx = np.arange(len(X))[valid]
        tr_i, te_i = train_test_split(
            idx, test_size=args.test_size, stratify=strata_v[valid], random_state=args.seed
        )
        Xtr, Xte, ytr, yte = X[tr_i], X[te_i], y[tr_i], y[te_i]
        bins_tr, bins_te = bins[tr_i], bins[te_i]

    if args.clf == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=args.seed,
        )
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]
    elif args.clf == "xgb":
        # Gradient-boosted trees baseline on the identical 30-d spectral
        # features (stronger non-FM competitor than RF). class imbalance via
        # scale_pos_weight = n_neg/n_pos on the train split.
        from xgboost import XGBClassifier
        spw = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", random_state=args.seed,
        )
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]
    else:
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        clf.fit(scaler.transform(Xtr), ytr)
        y_pred = clf.predict(scaler.transform(Xte))
        y_score = clf.predict_proba(scaler.transform(Xte))[:, 1]

    overall = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_pos_train": int(ytr.sum()),
        "n_pos_test": int(yte.sum()),
        "f1": float(f1_score(yte, y_pred)),
        "accuracy": float(accuracy_score(yte, y_pred)),
        "auroc": float(roc_auc_score(yte, y_score)) if len(set(yte)) == 2 else None,
    }

    bins_ordered = [b.value for b in FieldSizeBin.ordered()]
    per_bin = {}
    for bin_name in bins_ordered:
        mask = (yte == 1) & (bins_te == bin_name)
        if mask.sum() == 0:
            per_bin[bin_name] = {"n": 0, "recall": None}
            continue
        per_bin[bin_name] = {
            "n": int(mask.sum()),
            "recall": float(recall_score(yte[mask], y_pred[mask], zero_division=0)),
        }

    out = {
        "classifier": args.clf,
        "feature_set": f"30d_per_pixel_window{args.half_px*2+1}",
        "dataset": {
            "n_total": int(len(meta)),
            "n_positive": int((meta.label == 1).sum()),
            "n_negative": int((meta.label == 0).sum()),
            "test_size": args.test_size,
            "seed": args.seed,
            "split": args.split,
            "n_train_chips": n_train_chips,
            "n_test_chips": n_test_chips,
            "chip_overlap": chip_overlap,
        },
        "overall": overall,
        "per_pixel_recall_by_polygon_size": per_bin,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    banner("Per-pixel non-FM baseline results")
    log.info("F1=%.3f  acc=%.3f  AUROC=%s", overall["f1"], overall["accuracy"],
             f"{overall['auroc']:.3f}" if overall["auroc"] is not None else "n/a")
    for bin_name, rec in per_bin.items():
        r_str = f"{rec['recall']:.3f}" if rec["recall"] is not None else "n/a"
        log.info("  %-12s n=%5d  recall=%s", bin_name, rec["n"], r_str)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
