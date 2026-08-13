#!/usr/bin/env python
"""Build a chip index over Fields-of-The-World (FTW) footprints for our S2 pull.

For each FTW chip (aoi_id) we take its footprint centroid and select the
least-cloud Sentinel-2 L2A scene inside the FTW acquisition window, so the
pulled S2 is contemporaneous with the FTW labels. Output schema
matches data/index/*.jsonl plus an `aoi_id` so the FTW true-field polygons
(scripts/ftw_to_polygons.py) map 1:1 onto the pulled chips.

    python scripts/build_ftw_index.py --country india --limit 800 \
        --out data/index/ftw_india.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from shapely import wkb

# FTW acquisition windows (from data/ftw/data_config_<country>.json). window_a
# is the main growing-season window we pull.
WINDOWS = {
    "india":   ("2016-07-01", "2016-11-30"),
    "kenya":   ("2017-01-01", "2017-12-31"),  # widen; refined from config at runtime
    "vietnam": ("2021-07-01", "2021-09-30"),  # FTW window_a (main labeled season)
    "france":  ("2020-03-15", "2020-06-30"),  # FTW window_a (large-field negative control)
    "netherlands": ("2022-03-01", "2022-06-30"),  # FTW window_a (2nd large-field control)
}
GSD_M = 10.0
CHIP_PX = 256


def ee_init():
    try:
        import os
        proj = os.getenv("EE_PROJECT")
        ee.Initialize(project=proj) if proj else ee.Initialize()
    except Exception:
        ee.Initialize()


def least_cloud_scene(lon: float, lat: float, start: str, end: str, max_cloud: float = 60.0):
    pt = ee.Geometry.Point([lon, lat])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(pt).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .sort("CLOUDY_PIXEL_PERCENTAGE"))
    img = ee.Image(col.first())
    try:
        info = img.getInfo()
    except Exception:
        return None
    if info is None:
        return None
    props = info.get("properties", {})
    return {"scene_id": info.get("id", ""),
            "scene_date": str(props.get("system:time_start", "")),
            "cloud_pct": float(props.get("CLOUDY_PIXEL_PERCENTAGE", -1.0))}


def centroids(country: str) -> list[tuple[str, float, float]]:
    df = pd.read_parquet(Path("data/ftw") / f"chips_{country}.parquet")
    out = []
    for _, r in df.iterrows():
        g = r["geometry"]
        geom = wkb.loads(bytes(g)) if isinstance(g, (bytes, bytearray)) else g
        c = geom.centroid
        out.append((r["aoi_id"], float(c.x), float(c.y)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="india")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-cloud", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = args.out or Path(f"data/index/ftw_{args.country}.jsonl")

    start, end = WINDOWS.get(args.country, ("2016-01-01", "2016-12-31"))
    cents = centroids(args.country)
    rng = np.random.default_rng(20260514)
    if args.limit and len(cents) > args.limit:
        idx = rng.choice(len(cents), size=args.limit, replace=False)
        cents = [cents[i] for i in idx]

    ee_init()
    cc = {"country": args.country.upper()[:3]}

    def work(item):
        aoi, lon, lat = item
        for attempt in range(3):
            sc = least_cloud_scene(lon, lat, start, end, args.max_cloud)
            if sc and sc["scene_id"]:
                return {"aoi_id": aoi, "district": f"ftw_{aoi.split('_')[0]}",
                        "country": cc["country"], "lon": lon, "lat": lat,
                        "chip_size_px": CHIP_PX, "gsd_m": GSD_M,
                        "scene_id": sc["scene_id"], "scene_date": sc["scene_date"],
                        "cloud_pct": sc["cloud_pct"], "season_window": f"ftw_{args.country}_{start[:4]}"}
            time.sleep(1.0)
        return None

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, cents):
            done += 1
            if r:
                rows.append(r)
            if done % 50 == 0:
                print(f"  {done}/{len(cents)} processed, {len(rows)} kept")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}: {len(rows)}/{len(cents)} chips with a cloud-free {start[:4]} scene")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
