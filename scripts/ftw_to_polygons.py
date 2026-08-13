#!/usr/bin/env python
"""Convert FTW instance masks -> true-field polygons in our POLYGON_COLUMNS
format (drop-in replacement for WorldCover-derived polygons).

Each FTW field (unique instance ID) becomes one polygon, with area in m^2
(reprojected to the geometry's true local UTM) and the same size_bin scheme as
`extract_field_polygons.py`. Output is a geoparquet keyed on `chip_id`
(= FTW aoi_id), so the existing per-pixel / per-polygon extractors can consume
FTW exactly like a normal region once S2 is pulled over the same footprints.

    python scripts/ftw_to_polygons.py --country india \
        --out data/labels/polygons_ftw_india.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import transform_geom
from shapely.geometry import shape

from ftw_eval.data.labels import FieldSizeBin, POLYGON_COLUMNS

FTW = Path("data/ftw")
MIN_AREA, MAX_AREA = 100.0, 2_000_000.0


def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6) + 1
    band = 326 if lat >= 0 else 327
    return f"EPSG:{band}{zone:02d}"


def polygons_for_chip(mask_path: Path, aoi_id: str) -> list[dict]:
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        crs = src.crs
        tr = src.transform
    if arr.max() == 0:
        return []
    # relabel uint64 instance IDs -> sequential int32 (values irrelevant; each
    # field keeps a unique label, so per-field connectivity is preserved).
    _, inv = np.unique(arr, return_inverse=True)
    lab = inv.reshape(arr.shape).astype(np.int32)
    recs, pid = [], 0
    for geom, _val in features.shapes(lab, mask=(arr > 0), transform=tr):
        wgs = shape(transform_geom(str(crs), "EPSG:4326", geom))
        cx, cy = wgs.centroid.x, wgs.centroid.y
        utm = utm_epsg(cx, cy)
        area = float(shape(transform_geom("EPSG:4326", utm, wgs.__geo_interface__)).area)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        recs.append({
            "chip_id": aoi_id, "district": f"ftw_{aoi_id.split('_')[0]}",
            "polygon_id": f"{aoi_id}_{pid}", "area_m2": area,
            "size_bin": FieldSizeBin.from_area_m2(area).value,
            "centroid_lon": cx, "centroid_lat": cy, "geometry": wgs,
        })
        pid += 1
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="india")
    ap.add_argument("--ftw-dir", type=Path, default=FTW,
                    help="dir holding chips_<country>.parquet + label_masks/instance")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"data/labels/polygons_ftw_{args.country}.parquet")

    chips = pd.read_parquet(args.ftw_dir / f"chips_{args.country}.parquet")
    aoi_ids = list(chips["aoi_id"])
    if args.limit:
        aoi_ids = list(np.random.default_rng(20260514).choice(aoi_ids, size=min(args.limit, len(aoi_ids)), replace=False))
    inst = args.ftw_dir / "label_masks" / "instance"
    recs = []
    used = 0
    for aoi in aoi_ids:
        p = inst / f"{aoi}.tif"
        if p.exists():
            recs.extend(polygons_for_chip(p, aoi))
            used += 1
    if not recs:
        print("no polygons extracted"); return 1
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")[POLYGON_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out)
    bins = gdf["size_bin"].value_counts().reindex([b.value for b in FieldSizeBin.ordered()]).fillna(0).astype(int)
    print(f"wrote {out}: {len(gdf)} true-field polygons from {used} FTW chips")
    print("size bins:", bins.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
