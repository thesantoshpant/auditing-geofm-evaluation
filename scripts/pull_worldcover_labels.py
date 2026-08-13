#!/usr/bin/env python
"""Pull ESA WorldCover 2021 labels aligned to each chip.

For every row in the chip index, fetches the WorldCover v200 land-cover map
at 10 m, clipped to the chip's footprint, and writes it next to the imagery:
    data/chips/{district}/{chip_id}_worldcover.tif   # uint8, raw class codes

WorldCover class codes (ESA spec):
    10 = Tree cover           40 = Cropland         80 = Permanent water
    20 = Shrubland            50 = Built-up         90 = Herbaceous wetland
    30 = Grassland            60 = Bare/sparse veg  95 = Mangroves
                              70 = Snow and ice     100 = Moss and lichen

Binary cropland mask: pixel == 40. Computed at use time, not baked in here,
so we keep the option to use multi-class labels later.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import requests

from ftw_eval.utils.logging import banner, get_logger


log = get_logger("pull-worldcover")

DEFAULT_INDEX = Path("data/index/canonical.jsonl")
DEFAULT_OUT_DIR = Path("data/chips")
WORLDCOVER_ASSET = "ESA/WorldCover/v200/2021"


def utm_epsg_for_chip(lon: float, lat: float = 27.0) -> str:
    """Region-general UTM EPSG (zone from lon, hemisphere from lat).
    Name kept for back-compat; works for N and S hemispheres."""
    zone = int((lon + 180.0) // 6) + 1
    band = 326 if lat >= 0 else 327
    return f"EPSG:{band}{zone:02d}"


def chip_id(row: dict) -> str:
    date_part = row["scene_date"][:10] if len(row["scene_date"]) >= 10 else "unknown"
    return f"{row['lon']:.5f}_{row['lat']:.5f}_{date_part}"


def chip_geom(row: dict) -> ee.Geometry:
    half_m = (row["chip_size_px"] * row["gsd_m"]) / 2.0
    return ee.Geometry.Point([row["lon"], row["lat"]]).buffer(half_m).bounds()


def fetch_worldcover(row: dict, out: Path) -> int:
    img = ee.Image(WORLDCOVER_ASSET).select("Map")
    crs = utm_epsg_for_chip(row["lon"], row["lat"])  # fail-hard if lat missing (UTM hemisphere) 
    geom = chip_geom(row)
    url = img.getDownloadURL({
        "scale": row["gsd_m"],
        "region": geom,
        "format": "GEO_TIFF",
        "crs": crs,
    })
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return len(r.content)


def process_row(row: dict, out_dir: Path) -> dict:
    district_dir = out_dir / row["district"]
    cid = chip_id(row)
    wc_path = district_dir / f"{cid}_worldcover.tif"
    record = {"chip_id": cid, "ok": True, "skipped": False}
    if wc_path.exists():
        record["skipped"] = True
        return record
    try:
        n_bytes = fetch_worldcover(row, wc_path)
        record["bytes"] = n_bytes
    except Exception as e:
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        log.error("[%s] %s", cid, record["error"])
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--project", default=os.getenv("EE_PROJECT"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index.exists():
        log.error("Index not found at %s. Run build_chip_index.py first.", args.index)
        return 2

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    banner(f"Pulling WorldCover labels for {len(rows)} chips")

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_row, r, args.out_dir) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["skipped"]:
                skipped += 1
            elif r["ok"]:
                done += 1
            else:
                failed += 1
            if i % 25 == 0:
                log.info("Progress %d/%d  done=%d skipped=%d failed=%d",
                         i, len(rows), done, skipped, failed)

    banner(f"Complete: {done} new, {skipped} already-on-disk, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
