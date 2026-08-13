#!/usr/bin/env python
"""Per-pixel linear-probe eval. Companion to extract_features_per_pixel.py.

Resolves the n_tokens-asymmetry artifact: under per-pixel evaluation every
(feature, label) pair has the same "n_tokens=1" representation regardless
of polygon size, so the linear / non-linear head cannot exploit pool-size
differences between positives and negatives.

For each FM:
    1. Load features [N_samples, D] + meta parquet.
    2. Stratified train/test split by (district x label x size_bin).
    3. Logistic regression with class-balanced weights.
    4. Report:
        - Per-pixel F1 / accuracy / AUROC
        - Per-pixel recall stratified by polygon-size-of-containing-polygon
          (the size effect axis -- pixels inside small polygons should
          have lower recall under per-polygon protocol; checking whether
          this persists under per-pixel)
        - Per-polygon recall via per-pixel majority vote (aggregate
          predictions for all pixels with the same polygon_id, threshold
          at 0.5 of cropland predictions).
        - Compare to per-polygon baseline numbers when --compare-baseline
          is set.

Usage:
    python scripts/eval_per_pixel.py \
        --features-dir data/features_per_pixel \
        --out data/results/eval_per_pixel_canonical.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.getLogger("rasterio").setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ftw_eval.data.labels import FieldSizeBin  # noqa: E402
from ftw_eval.utils.logging import banner, get_logger  # noqa: E402


log = get_logger("eval-pp-pix")


def build_stratum(row) -> str:
    if row["label"] == 0:
        return f"neg_{row['district']}"
    return f"pos_{row['district']}_{row['size_bin']}"


def per_polygon_majority_vote(
    y_true: np.ndarray, y_pred: np.ndarray, polygon_ids: np.ndarray
) -> dict:
    """Group predictions by polygon_id (string or int), return
    {pid: (true_label, frac_pred_pos)}."""
    out: dict = {}
    by_pid: dict = defaultdict(list)
    by_pid_true: dict = {}
    for yt, yp, pid in zip(y_true, y_pred, polygon_ids):
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        by_pid[pid].append(int(yp))
        by_pid_true[pid] = int(yt)
    for pid, preds in by_pid.items():
        out[pid] = (by_pid_true[pid], float(np.mean(preds)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fms", nargs="*", default=None,
                   help="FMs to evaluate (default: discover from features_*.npy)")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--split", choices=["chip", "pixel"], default="chip",
                   help="'chip' (default): group all pixels of a chip into one "
                        "of train/test (no spatial leakage). 'pixel': legacy "
                        "leaky per-pixel split, kept only for ablation.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    meta = pd.read_parquet(args.features_dir / "features_per_pixel_meta.parquet")
    banner(f"Per-pixel eval on {len(meta)} samples")
    log.info("  positives=%d  negatives=%d",
             int((meta.label == 1).sum()), int((meta.label == 0).sum()))
    log.info("  districts=%s", sorted(meta.district.unique()))

    if args.fms is None:
        fms = sorted([
            f.stem.replace("features_", "").replace("_", "-")
            for f in args.features_dir.glob("features_*.npy")
        ])
        fms = [f for f in fms if not f.endswith("meta")]
    else:
        fms = args.fms
    log.info("  evaluating %d FMs: %s", len(fms), fms)

    y = meta["label"].to_numpy()
    strata = meta.apply(build_stratum, axis=1)
    counts = strata.value_counts()
    keep = strata.isin(counts[counts >= 2].index).to_numpy()

    sample_ids = meta["sample_id"].to_numpy()
    polygon_ids = meta["polygon_id"].to_numpy()
    size_bins = meta["size_bin"].to_numpy()
    districts = meta["district"].to_numpy()
    areas = meta["area_m2"].to_numpy()
    chip_ids = meta["chip_id"].to_numpy()

    keep_idx = np.arange(len(meta))[keep]
    if args.split == "chip":
        # CHIP-GROUPED split: all pixels from a given chip go to exactly one
        # of train/test. This is REQUIRED to avoid spatial-autocorrelation
        # leakage -- pixels in one 256-px chip are highly correlated and many
        # share byte-identical FM tokens, so a pixel-level split lets the
        # classifier memorize test pixels from their train neighbours.
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_pos, te_pos = next(gss.split(keep_idx, y[keep], groups=chip_ids[keep]))
        idx_tr = keep_idx[tr_pos]
        idx_te = keep_idx[te_pos]
        n_chip_tr = len(set(chip_ids[idx_tr]))
        n_chip_te = len(set(chip_ids[idx_te]))
        overlap = set(chip_ids[idx_tr]) & set(chip_ids[idx_te])
        log.info("  CHIP-GROUPED split: train chips=%d test chips=%d overlap=%d",
                 n_chip_tr, n_chip_te, len(overlap))
    else:
        # Legacy pixel-level split (leaky; kept only for ablation/comparison).
        idx_tr, idx_te = train_test_split(
            keep_idx, test_size=args.test_size,
            stratify=strata[keep].to_numpy(), random_state=args.seed,
        )
    log.info("  split=%s  train=%d  test=%d  (test pos=%d, test neg=%d)",
             args.split, len(idx_tr), len(idx_te), int(y[idx_te].sum()),
             int((y[idx_te] == 0).sum()))

    results = {
        "dataset": {
            "n_total": int(len(meta)),
            "n_positive": int((meta.label == 1).sum()),
            "n_negative": int((meta.label == 0).sum()),
            "districts": sorted(meta.district.unique().tolist()),
            "test_size": args.test_size,
            "seed": args.seed,
            "split": args.split,
            "n_train_chips": int(len(set(chip_ids[idx_tr]))),
            "n_test_chips": int(len(set(chip_ids[idx_te]))),
            "chip_overlap": int(len(set(chip_ids[idx_tr]) & set(chip_ids[idx_te]))),
        },
        "fms": {},
    }

    bins_ordered = [b.value for b in FieldSizeBin.ordered()]

    for fm_name in fms:
        npy_name = f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"
        npy_path = args.features_dir / npy_name
        if not npy_path.exists():
            # try alternate naming (anysat -> any-sat collapse, etc)
            alt = list(args.features_dir.glob(f"features_*{fm_name.split('-')[0]}*.npy"))
            if not alt:
                log.warning("[%s] no feature file at %s, skipping", fm_name, npy_path)
                continue
            npy_path = alt[0]
        X = np.load(npy_path)
        if len(X) != len(meta):
            log.warning("[%s] feature rows %d != meta rows %d, skipping",
                        fm_name, len(X), len(meta))
            continue

        # Drop all-zero rows: the extractor preallocates zeros and a failed
        # FM forward leaves a chip's rows at zero. Including them would feed
        # spurious zero "features" to the classifier. Restrict train/test to
        # valid (non-all-zero) rows.
        valid = ~np.all(X == 0, axis=1)
        n_invalid = int((~valid).sum())
        if n_invalid:
            log.warning("[%s] dropping %d all-zero feature rows (failed forwards)",
                        fm_name, n_invalid)
        fm_tr = idx_tr[valid[idx_tr]]
        fm_te = idx_te[valid[idx_te]]

        scaler = StandardScaler().fit(X[fm_tr])
        Xtr = scaler.transform(X[fm_tr])
        Xte = scaler.transform(X[fm_te])
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=args.seed
        )
        clf.fit(Xtr, y[fm_tr])
        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]

        overall = {
            "n_train": int(len(fm_tr)),
            "n_test": int(len(fm_te)),
            "n_invalid_dropped": n_invalid,
            "n_pos_test": int(y[fm_te].sum()),
            "f1": float(f1_score(y[fm_te], y_pred)),
            "accuracy": float(accuracy_score(y[fm_te], y_pred)),
            "auroc": float(roc_auc_score(y[fm_te], y_score)) if len(set(y[fm_te])) == 2 else None,
        }

        # Per-pixel recall stratified by polygon-size-of-containing-polygon
        per_bin = {}
        for bin_name in bins_ordered:
            mask = (y[fm_te] == 1) & (size_bins[fm_te] == bin_name)
            if mask.sum() == 0:
                per_bin[bin_name] = {"n": 0, "recall": None}
                continue
            per_bin[bin_name] = {
                "n": int(mask.sum()),
                "recall": float(recall_score(y[fm_te][mask], y_pred[mask], zero_division=0)),
            }
        # Target pixels not covered by any polygon (accept both legacy
        # "uncovered_crop" and new "uncovered_target" labels)
        mask_uncov = (y[fm_te] == 1) & (
            (size_bins[fm_te] == "uncovered_target")
            | (size_bins[fm_te] == "uncovered_crop")
        )
        per_bin["uncovered_target"] = {
            "n": int(mask_uncov.sum()),
            "recall": float(recall_score(y[fm_te][mask_uncov], y_pred[mask_uncov], zero_division=0))
            if mask_uncov.sum() > 0 else None,
        }

        # Per-polygon majority vote (test set only, polygons with >= 3 test pixels)
        pid_te = polygon_ids[fm_te]
        per_poly = per_polygon_majority_vote(y[fm_te], y_pred, pid_te)
        # Threshold at 0.5
        per_polygon_recall_by_bin = {}
        meta_test = meta.iloc[fm_te].reset_index(drop=True)
        # For per-polygon recall by bin, build {pid -> bin}
        pid_to_bin = {}
        for _, row in meta_test.iterrows():
            pid = row["polygon_id"]
            if pid is None or (isinstance(pid, float) and np.isnan(pid)):
                continue
            pid_to_bin[pid] = row["size_bin"]
        bin_to_pids = defaultdict(list)
        for pid, b in pid_to_bin.items():
            if pid in per_poly:
                bin_to_pids[b].append(pid)
        for bin_name in bins_ordered:
            pids = bin_to_pids.get(bin_name, [])
            if not pids:
                per_polygon_recall_by_bin[bin_name] = {"n_polygons": 0, "recall": None}
                continue
            # A polygon is a "true positive" if frac of cropland predictions >= 0.5
            preds = [1 if per_poly[pid][1] >= 0.5 else 0 for pid in pids]
            per_polygon_recall_by_bin[bin_name] = {
                "n_polygons": len(pids),
                "recall": float(np.mean(preds)),
            }

        results["fms"][fm_name] = {
            "overall": overall,
            "per_pixel_recall_by_polygon_size": per_bin,
            "per_polygon_recall_by_size_majority_vote": per_polygon_recall_by_bin,
        }
        log.info("[%s] F1=%.3f  acc=%.3f  AUROC=%.3f",
                 fm_name, overall["f1"], overall["accuracy"], overall["auroc"] or -1)
        for bin_name in bins_ordered:
            pr = per_bin[bin_name]
            ppr = per_polygon_recall_by_bin.get(bin_name, {})
            r1 = f"{pr['recall']:.3f}" if pr["recall"] is not None else "n/a"
            r2 = f"{ppr['recall']:.3f}" if ppr.get("recall") is not None else "n/a"
            log.info("    %-12s n_px=%5d  pix-recall=%s   n_poly=%4d  poly-recall=%s",
                     bin_name, pr["n"], r1,
                     ppr.get("n_polygons", 0), r2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
