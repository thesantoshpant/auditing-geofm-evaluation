#!/usr/bin/env python
"""Proxy-sensitivity analysis: TRUE field sizes (Fields of The World instance
masks) vs our WorldCover cropland-patch proxy sizes.

FTW instance masks give one unique integer ID per real agricultural field.
We vectorize them, reproject to local UTM, and compute true field areas, then
compare the size distribution to our WorldCover-derived "cropland-patch" sizes.
This quantifies how much the WorldCover proxy distorts the size axis that the
benchmark's central finding rests on.

    python scripts/ftw_field_sizes.py --country india --limit 600
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import transform_geom
from shapely.geometry import shape

FTW = Path("data/ftw")
MIN_AREA, MAX_AREA = 100.0, 2_000_000.0  # Shared polygon-area filter.
BINS = [("<0.1ha", 0, 1e3), ("0.1-0.3ha", 1e3, 3e3), ("0.3-0.5ha", 3e3, 5e3),
        ("0.5-1ha", 5e3, 1e4), (">1ha", 1e4, np.inf)]


def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6) + 1
    band = 326 if lat >= 0 else 327
    return f"EPSG:{band}{zone:02d}"


def bin_of(a: float) -> str:
    for name, lo, hi in BINS:
        if lo <= a < hi:
            return name
    return ">1ha"


def field_areas_for_chip(mask_path: Path) -> list[float]:
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        crs = src.crs
        tr = src.transform
    if arr.max() == 0:
        return []
    # Instance IDs are uint64 and can exceed int32 (e.g. Kenya); features.shapes
    # only accepts int16/32/uint8/16/float. Relabel to sequential int32; the ID
    # values are irrelevant, only per-field connectivity matters (each FTW field
    # already has a unique ID, so distinct fields stay distinct after relabel).
    _, inv = np.unique(arr, return_inverse=True)
    lab = inv.reshape(arr.shape).astype(np.int32)
    areas = []
    # mask=(arr>0) restricts shapes to foreground, so every emitted region is a field.
    for geom, _val in features.shapes(lab, mask=(arr > 0), transform=tr):
        wgs = shape(transform_geom(str(crs), "EPSG:4326", geom))
        c = wgs.centroid
        utm = utm_epsg(c.x, c.y)
        ga = shape(transform_geom("EPSG:4326", utm, wgs.__geo_interface__))
        a = float(ga.area)
        if MIN_AREA <= a <= MAX_AREA:
            areas.append(a)
    return areas


def dist(areas: np.ndarray) -> dict:
    n = len(areas)
    counts = {name: int(((areas >= lo) & (areas < hi)).sum()) for name, lo, hi in BINS}
    return {
        "n": n,
        "median_ha": round(float(np.median(areas)) / 1e4, 4) if n else None,
        "mean_ha": round(float(np.mean(areas)) / 1e4, 4) if n else None,
        "frac_by_bin": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "count_by_bin": counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="india")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--wc-polygons", type=Path, default=Path("data/labels/polygons_canonical_e.parquet"),
                    help="our WorldCover-derived polygons to compare against")
    ap.add_argument("--out", type=Path, default=Path("data/results/ftw_proxy_sensitivity_india.json"))
    args = ap.parse_args()

    chips = pd.read_parquet(FTW / f"chips_{args.country}.parquet")
    aoi_ids = list(chips["aoi_id"])
    rng = np.random.default_rng(20260514)
    if args.limit and len(aoi_ids) > args.limit:
        aoi_ids = list(rng.choice(aoi_ids, size=args.limit, replace=False))

    inst_dir = FTW / "label_masks" / "instance"
    true_areas: list[float] = []
    used = 0
    for aoi in aoi_ids:
        p = inst_dir / f"{aoi}.tif"
        if not p.exists():
            continue
        true_areas.extend(field_areas_for_chip(p))
        used += 1
    true_areas = np.array(true_areas, dtype=float)

    out = {"country": args.country, "chips_used": used,
           "FTW_true_fields": dist(true_areas)}

    if args.wc_polygons.exists():
        wc = pd.read_parquet(args.wc_polygons)
        wc_areas = wc["area_m2"].to_numpy(dtype=float)
        wc_areas = wc_areas[(wc_areas >= MIN_AREA) & (wc_areas <= MAX_AREA)]
        out["WorldCover_proxy"] = dist(wc_areas)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
