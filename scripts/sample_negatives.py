#!/usr/bin/env python
"""Sample non-cropland negative patches for binary classification.

For each chip in the manifest, find pixels where WorldCover != cropland(40)
and sample N points uniformly. Each sampled point becomes one negative
example for the cropland-vs-non-cropland zero-shot eval.

Output: data/labels/negatives_canonical.parquet
        columns: negative_id, chip_id, district, center_lon, center_lat,
                 center_row_px, center_col_px, worldcover_class

Notes:
    - We enforce a minimum 20 px (200 m) spacing between negatives within
      the same chip to avoid clumping in a single non-cropland blob.
    - Negatives carry the WorldCover class at their center for diagnostic
      purposes (so we can later check which non-cropland classes are
      hardest to separate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from ftw_eval.data.labels import (
    WORLDCOVER_CLASSES,
    WORLDCOVER_CROPLAND_CODE,
    read_manifest,
)

from ftw_eval.utils.logging import banner, get_logger


log = get_logger("negatives")

DEFAULT_MANIFEST = Path("data/chips/manifest_canonical.jsonl")
DEFAULT_OUT = Path("data/labels/negatives_canonical.parquet")


def sample_chip_negatives(
    worldcover_path: Path,
    chip_id: str,
    district: str,
    target_n: int,
    min_spacing_px: int,
    rng: np.random.Generator,
    positive_class: int = WORLDCOVER_CROPLAND_CODE,
) -> list[dict]:
    with rasterio.open(worldcover_path) as src:
        wc = src.read(1)
        transform = src.transform
        crs = str(src.crs)

    non_positive = (wc != positive_class) & (wc != 0)
    rs, cs = np.where(non_positive)
    if len(rs) == 0:
        return []
    candidate_idx = rng.permutation(len(rs))
    chosen: list[tuple[int, int]] = []
    for i in candidate_idx:
        r, c = int(rs[i]), int(cs[i])
        if all((abs(r - r2) >= min_spacing_px or abs(c - c2) >= min_spacing_px)
               for r2, c2 in chosen):
            chosen.append((r, c))
            if len(chosen) >= target_n:
                break

    records = []
    for k, (r, c) in enumerate(chosen):
        x, y = rasterio.transform.xy(transform, r, c)  # native CRS
        lon_arr, lat_arr = warp_transform(crs, "EPSG:4326", [x], [y])
        records.append({
            "negative_id": f"{chip_id}_neg{k}",
            "chip_id": chip_id,
            "district": district,
            "center_lon": lon_arr[0],
            "center_lat": lat_arr[0],
            "center_row_px": r,
            "center_col_px": c,
            "worldcover_class": int(wc[r, c]),
            "worldcover_class_name": WORLDCOVER_CLASSES.get(int(wc[r, c]), "unknown"),
        })
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--per-chip", type=int, default=24,
                   help="target negatives per chip (default matches ~24 positives/chip)")
    p.add_argument("--min-spacing-px", type=int, default=20,
                   help="minimum pixel spacing between negatives within a chip")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--positive-class", type=int, default=WORLDCOVER_CROPLAND_CODE,
                   help="WorldCover code treated as the positive class; negatives "
                        "are sampled from pixels NOT in this class. Default 40 (cropland).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        log.error("Manifest not found at %s", args.manifest)
        return 2

    entries = read_manifest(args.manifest)
    banner(f"Sampling negatives from {len(entries)} chips")
    rng = np.random.default_rng(args.seed)

    all_records: list[dict] = []
    no_negatives = 0
    for i, entry in enumerate(entries, 1):
        if entry.worldcover_path is None or not Path(entry.worldcover_path).exists():
            continue
        recs = sample_chip_negatives(
            Path(entry.worldcover_path),
            entry.chip_id,
            entry.region.district,
            args.per_chip,
            args.min_spacing_px,
            rng,
            positive_class=args.positive_class,
        )
        if not recs:
            no_negatives += 1
        all_records.extend(recs)
        if i % 50 == 0:
            log.info("Processed %d/%d chips; running negatives = %d",
                     i, len(entries), len(all_records))

    log.info("Total negatives: %d (chips with no non-crop pixels: %d)",
             len(all_records), no_negatives)

    if not all_records:
        log.warning("No negatives sampled.")
        return 1

    df = pd.DataFrame(all_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info("Wrote %s (%d rows)", args.out, len(df))

    by_class = df["worldcover_class_name"].value_counts()
    banner("Negatives by WorldCover class")
    for k, v in by_class.items():
        log.info("  %-22s %6d", k, v)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
