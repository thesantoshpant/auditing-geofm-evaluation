#!/usr/bin/env python
"""Assemble a chip manifest from the index + downloaded files.

Walks ``data/chips/<district>/`` and joins each chip's S2/S1/WorldCover paths
to the corresponding index row, then writes ``ChipManifestEntry`` JSONL.

The manifest is the canonical input to every downstream script
(polygon extraction, splits, eval). Decoupling the manifest from the index
lets us skip chips whose download failed without polluting the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftw_eval.data.labels import ChipManifestEntry, RegionTag, write_manifest
from ftw_eval.utils.logging import banner, get_logger


log = get_logger("manifest")

DEFAULT_INDEX = Path("data/index/canonical.jsonl")
DEFAULT_CHIP_DIR = Path("data/chips")
DEFAULT_OUT = Path("data/chips/manifest_canonical.jsonl")


def chip_id_from_row(row: dict) -> str:
    date_part = row["scene_date"][:10] if len(row["scene_date"]) >= 10 else "unknown"
    return f"{row['lon']:.5f}_{row['lat']:.5f}_{date_part}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--require-s1", action="store_true",
                   help="exclude rows missing the S1 chip")
    p.add_argument("--require-worldcover", action="store_true",
                   help="exclude rows missing the WorldCover label")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index.exists():
        log.error("Index not found at %s", args.index)
        return 2

    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    banner(f"Assembling manifest from {len(rows)} index rows")

    entries: list[ChipManifestEntry] = []
    missing_s2 = missing_s1 = missing_wc = 0
    for row in rows:
        district = row["district"]
        cid = chip_id_from_row(row)
        district_dir = args.chip_dir / district
        s2_path = district_dir / f"{cid}_s2.tif"
        s1_path = district_dir / f"{cid}_s1.tif"
        wc_path = district_dir / f"{cid}_worldcover.tif"

        if not s2_path.exists():
            missing_s2 += 1
            continue
        if args.require_s1 and not s1_path.exists():
            missing_s1 += 1
            continue
        if args.require_worldcover and not wc_path.exists():
            missing_wc += 1
            continue

        entries.append(ChipManifestEntry(
            chip_id=cid,
            region=RegionTag(country=row.get("country", ""), province=None, district=district),
            lon=row["lon"], lat=row["lat"], gsd_m=row["gsd_m"],
            chip_size_px=row["chip_size_px"],
            scene_id=row["scene_id"], scene_date=row["scene_date"],
            cloud_pct=row["cloud_pct"],
            s2_path=s2_path,
            s1_path=s1_path if s1_path.exists() else None,
            worldcover_path=wc_path if wc_path.exists() else None,
        ))

    write_manifest(entries, args.out)
    log.info("Wrote %s (%d entries)", args.out, len(entries))
    log.info("Dropped: missing_s2=%d missing_s1=%d missing_worldcover=%d",
             missing_s2, missing_s1, missing_wc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
