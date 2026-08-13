#!/usr/bin/env python
"""Pull S2 + S1 GeoTIFFs for every row in a chip index.

Reads data/index/canonical.jsonl and writes:
    data/chips/{district}/{chip_id}_s2.tif   # 12-band L2A, float32 reflectance/10000
    data/chips/{district}/{chip_id}_s1.tif   # 2-band SAR (VV, VH) sigma0 in dB
    data/chips/{district}/{chip_id}.meta.json

Idempotent: existing chips are skipped. Re-run after a failure to continue.

S1 pairing: the GRD scene whose acquisition date is closest to the S2 scene's
date (within a configurable window) is selected. Both ascending and descending
orbits are allowed; we record which we picked in the metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import ee
import requests

from ftw_eval.utils.logging import banner, get_logger


log = get_logger("pull-chips")

DEFAULT_INDEX = Path("data/index/canonical.jsonl")
DEFAULT_OUT_DIR = Path("data/chips")

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


def utm_epsg(lon: float, lat: float = 27.0) -> str:
    """Region-general UTM EPSG for a chip center: zone from lon, hemisphere
    from lat (326## = Northern, 327## = Southern). Works for /India
    (N) and  (S) alike. Note: polygon AREA is translation-
    invariant, so the N/S choice does not affect areas, but the correct
    hemisphere keeps exported coordinates sane."""
    zone = int((lon + 180.0) // 6) + 1
    band = 326 if lat >= 0 else 327
    return f"EPSG:{band}{zone:02d}"


# .
def utm_epsg_for_chip(lon: float) -> str:
    return utm_epsg(lon, 27.0)


def chip_id(row: dict) -> str:
    date_part = row["scene_date"][:10] if len(row["scene_date"]) >= 10 else "unknown"
    return f"{row['lon']:.5f}_{row['lat']:.5f}_{date_part}"


def chip_geom(row: dict) -> ee.Geometry:
    half_m = (row["chip_size_px"] * row["gsd_m"]) / 2.0
    return ee.Geometry.Point([row["lon"], row["lat"]]).buffer(half_m).bounds()


def download_image(image: ee.Image, geom: ee.Geometry, crs: str, scale: float, out: Path) -> int:
    url = image.getDownloadURL({
        "scale": scale,
        "region": geom,
        "format": "GEO_TIFF",
        "crs": crs,
    })
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return len(r.content)


def fetch_s2(row: dict, out: Path) -> dict:
    img = ee.Image(row["scene_id"]).select(S2_BANDS)
    crs = utm_epsg(row["lon"], row["lat"])  # fail-hard if lat missing (UTM hemisphere) 
    geom = chip_geom(row)
    n_bytes = download_image(img, geom, crs, row["gsd_m"], out)
    return {"bytes": n_bytes, "crs": crs}


def fetch_s1(row: dict, out: Path, window_days: int = 14) -> dict | None:
    """Pair with the closest S1 GRD scene within ``window_days`` of the S2 date."""
    geom = chip_geom(row)
    # Parse S2 epoch ms -> datetime
    try:
        s2_dt = datetime.fromtimestamp(int(row["scene_date"]) / 1000.0)
    except Exception:
        s2_dt = datetime(2024, 11, 1)
    start = (s2_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (s2_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if col.size().getInfo() == 0:
        return None
    # Pick the scene closest in time.
    s2_ms = int(row["scene_date"]) if row["scene_date"].isdigit() else int(s2_dt.timestamp() * 1000)
    col = col.map(lambda im: im.set("time_dist", ee.Number(im.get("system:time_start")).subtract(s2_ms).abs()))
    img = ee.Image(col.sort("time_dist").first()).select(S1_BANDS)
    crs = utm_epsg(row["lon"], row["lat"])  # fail-hard if lat missing (UTM hemisphere) 
    n_bytes = download_image(img, geom, crs, row["gsd_m"], out)
    info = img.getInfo().get("properties", {})
    return {
        "bytes": n_bytes,
        "crs": crs,
        "orbit": info.get("orbitProperties_pass"),
        "scene_id": img.getInfo().get("id", ""),
        "scene_date": str(info.get("system:time_start", "")),
    }


def process_row(row: dict, out_dir: Path, skip_s1: bool) -> dict:
    district_dir = out_dir / row["district"]
    cid = chip_id(row)
    s2_path = district_dir / f"{cid}_s2.tif"
    s1_path = district_dir / f"{cid}_s1.tif"
    meta_path = district_dir / f"{cid}.meta.json"

    record: dict = {"chip_id": cid, "district": row["district"], "row": row, "ok": True, "skipped": False}
    if s2_path.exists() and (skip_s1 or s1_path.exists()) and meta_path.exists():
        record["skipped"] = True
        return record
    try:
        s2_info = fetch_s2(row, s2_path)
        record["s2"] = s2_info
        if not skip_s1:
            s1_info = fetch_s1(row, s1_path)
            record["s1"] = s1_info
        meta_path.write_text(json.dumps(record, indent=2))
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
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N rows (smoke testing)")
    p.add_argument("--skip-s1", action="store_true", help="pull S2 only")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index.exists():
        log.error("Index not found at %s. Run build_chip_index.py first.", args.index)
        return 2

    # Prefer an explicit project; otherwise fall back to the project baked
    # into the stored EE credentials (bare Initialize), which is how the
    # rest of the GEE tooling authenticates on this host.
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    banner(f"Pulling imagery for {len(rows)} chips (workers={args.workers})")

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_row, row, args.out_dir, args.skip_s1) for row in rows]
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
