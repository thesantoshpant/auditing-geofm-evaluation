#!/usr/bin/env python
"""Per-pixel feature extraction: upsample FM token grid to chip pixels.

Resolves the n_tokens-asymmetry methodology artifact identified in .
Per the per-polygon protocol, positives draw variable-size token pools and
negatives draw a fixed ~16x16 pool; any expressive classifier exploits the
difference. The per-pixel protocol replaces polygon-level aggregation with
single-token-per-pixel sampling: every (feature, label) pair has the same
"n_tokens = 1" representation.

For each chip:
    1. Read S2 GeoTIFF + WorldCover GeoTIFF (same chip footprint).
    2. Run each FM, get tokens [N, D], reshape to [H_tok, W_tok, D].
    3. Build a per-chip-pixel feature lookup via nearest-neighbor token
       index mapping (chip pixel r,c -> token tr,tc by linear scale).
    4. Build a per-chip-pixel polygon-id raster by rasterizing the
       chip's positive polygons (so we can stratify per-pixel recall
       by the size of the polygon a cropland pixel belongs to).
    5. Sample N positive (cropland) and N negative (non-cropland)
       chip pixels with a chip-seeded RNG (the same pixels for all FMs).
    6. Gather features + labels + polygon_id + district + chip_id.

Output:
    data/features_per_pixel_<region>/features_<fm>.npy shape [N_samples, D]
    data/features_per_pixel_<region>/features_per_pixel_meta.parquet
        columns: sample_id, chip_id, district, label, polygon_id (or NaN),
        pixel_r, pixel_c, area_m2_of_polygon (or NaN), size_bin (or non_crop)

AnySat is excluded by default -- it returns a tile-level feature only and
upsampling to per-pixel would replicate the same vector across all chip
pixels, defeating the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CPL_LOG_ERRORS", "OFF")
os.environ.setdefault("CPL_DEBUG", "OFF")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Photometric.*")

import geopandas as gpd # noqa: E402
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import rasterio # noqa: E402
import torch # noqa: E402
from rasterio.features import rasterize # noqa: E402
from rasterio.warp import transform_geom # noqa: E402
from shapely.geometry import shape as shapely_shape # noqa: E402

from ftw_eval.data.chip import Chip, S2_BAND_ORDER # noqa: E402
from ftw_eval.data.labels import ( # noqa: E402
    WORLDCOVER_CROPLAND_CODE,
    read_manifest,
)
from ftw_eval.data.token_pool import reshape_to_grid # noqa: E402
from ftw_eval.model_zoo import get_model # noqa: E402
from ftw_eval.utils.logging import banner, get_logger # noqa: E402

log = get_logger("features-pp-pix")

DEFAULT_MANIFEST = Path("data/chips/manifest_canonical.jsonl")
DEFAULT_POLYGONS = Path("data/labels/polygons_canonical.parquet")
DEFAULT_OUT_DIR = Path("data/features_per_pixel")

FM_NAMES = ["clay-v1", "prithvi-eo-2.0-300m", "terramind-v1-base"]


def _parse_scene_date(scene_date) -> datetime:
    """Manifest scene_date is epoch-millis (str/int). Fall back to a
    fixed date only if parsing fails."""
    try:
        ts = int(str(scene_date)) / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return datetime(2024, 11, 1, tzinfo=timezone.utc)


def chip_from_geotiff_array(s2_path: Path, scene_date=None) -> tuple[Chip, int, int]:
    with rasterio.open(s2_path) as src:
        arr = src.read().astype(np.float32) / 10_000.0
        h, w = src.height, src.width
        gsd = abs(src.transform.a)
        cy, cx = h // 2, w // 2
        x, y = src.xy(cy, cx)
        from rasterio.warp import transform as warp_transform
        lons, lats = warp_transform(src.crs, "EPSG:4326", [x], [y])
        chip = Chip(
            s2=arr,
            s2_bands=list(S2_BAND_ORDER[: arr.shape[0]]),
            lat=lats[0], lon=lons[0], gsd_m=gsd,
            date=_parse_scene_date(scene_date),
        )
    return chip, h, w


def build_polygon_id_raster(
    chip_path: Path, polygons_for_chip: list[dict]
) -> tuple[np.ndarray, dict[int, dict]]:
    """Burn polygon int-ids onto a chip-shaped raster. Returns
    (id_raster, id_to_polygon_meta). Zero in the raster means "no polygon
    covers this pixel"; the int-id starts at 1 and is local to this chip.
    """
    if not polygons_for_chip:
        with rasterio.open(chip_path) as src:
            return np.zeros((src.height, src.width), dtype=np.int32), {}

    with rasterio.open(chip_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        target_crs = str(src.crs)

    shapes = []
    id_to_meta = {}
    for i, p in enumerate(polygons_for_chip, start=1):
        poly_proj_geo = transform_geom("EPSG:4326", target_crs, p["geometry"].__geo_interface__)
        shapes.append((shapely_shape(poly_proj_geo), i))
        id_to_meta[i] = p
    arr = rasterize(
        shapes, out_shape=out_shape, fill=0, transform=transform, dtype=np.int32
    )
    return arr, id_to_meta


def build_token_index_maps(
    chip_h: int, chip_w: int, h_tok: int, w_tok: int,
    fm_input_size_px: int, fm_patch_size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-chip-pixel token indices (row_idx[H], col_idx[W]).

    A chip pixel r maps to FM input pixel r_fm = r * (fm_input/chip),
    which maps to token tr = r_fm // patch_size. This collapses to
    tr = (r * h_tok) // chip_h when fm_input = h_tok * patch_size,
    which is true for every FM here.
    """
    row_scale = h_tok / float(chip_h)
    col_scale = w_tok / float(chip_w)
    row_idx = np.clip((np.arange(chip_h) * row_scale).astype(np.int64), 0, h_tok - 1)
    col_idx = np.clip((np.arange(chip_w) * col_scale).astype(np.int64), 0, w_tok - 1)
    _ = fm_input_size_px, fm_patch_size_px # kept for future non-linear mappings
    return row_idx, col_idx


def _gather_multiscale(
    tokens_2d: np.ndarray, tr: np.ndarray, tc: np.ndarray,
    dilations: tuple[int, ...] = (0, 1, 3),
) -> np.ndarray:
    """For each (tr, tc), gather the covering token + mean pools over
    dilated neighborhoods. Output shape: (len(tr), D * len(dilations)).

    k=0 is just the covering token (identical to non-multiscale path).
    k>=1 averages over (2k+1) x (2k+1) token window centered on (tr, tc),
    with boundary clipping.
    """
    h_tok, w_tok, d = tokens_2d.shape
    n = len(tr)
    out = np.zeros((n, d * len(dilations)), dtype=np.float32)
    for di, k in enumerate(dilations):
        start = di * d
        end = start + d
        if k == 0:
            out[:, start:end] = tokens_2d[tr, tc].astype(np.float32)
            continue
        for i in range(n):
            r0 = max(0, int(tr[i]) - k)
            c0 = max(0, int(tc[i]) - k)
            r1 = min(h_tok - 1, int(tr[i]) + k)
            c1 = min(w_tok - 1, int(tc[i]) + k)
            chunk = tokens_2d[r0:r1 + 1, c0:c1 + 1]
            out[i, start:end] = chunk.reshape(-1, d).mean(axis=0)
    return out


def sample_pixels_for_chip(
    cropland_mask: np.ndarray, n_per_class: int, rng: np.random.Generator,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows, cols, labels) sampled from a chip. Up to n_per_class
    cropland pixels and n_per_class non-cropland pixels.

    valid_mask (optional) excludes S2 no-data / all-zero pixels from BOTH
    classes so the negative pool cannot be padded with no-data fill. 
    """
    if valid_mask is None:
        valid_mask = np.ones(cropland_mask.shape, dtype=bool)
    pos_idx = np.argwhere((cropland_mask == 1) & valid_mask)
    neg_idx = np.argwhere((cropland_mask == 0) & valid_mask)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return (np.empty(0, dtype=np.int64),) * 3
    n_pos = min(n_per_class, len(pos_idx))
    n_neg = min(n_per_class, len(neg_idx))
    pos_sel = pos_idx[rng.choice(len(pos_idx), size=n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(len(neg_idx), size=n_neg, replace=False)]
    rows = np.concatenate([pos_sel[:, 0], neg_sel[:, 0]]).astype(np.int64)
    cols = np.concatenate([pos_sel[:, 1], neg_sel[:, 1]]).astype(np.int64)
    labels = np.concatenate([
        np.ones(n_pos, dtype=np.int8),
        np.zeros(n_neg, dtype=np.int8),
    ])
    return rows, cols, labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--polygons", type=Path, default=DEFAULT_POLYGONS,
                   help="polygons parquet (for polygon_id stratification)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fms", nargs="*", default=FM_NAMES,
                   help="FMs to extract (excludes anysat by default; tile-only)")
    p.add_argument("--n-per-class", type=int, default=200,
                   help="pixels sampled per (chip, label) class")
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-chips", type=int, default=None,
                   help="optional cap for quick sanity")
    p.add_argument("--multiscale", action="store_true",
                   help="per-pixel TokenPool: concat covering token with "
                        "mean pools of k=1 and k=3 token-radius neighborhoods "
                        "(output dim = 3 * base FM dim)")
    p.add_argument("--target-class", type=int, default=None,
                   help="WorldCover class code for the positive class "
                        "(default: cropland=40). Use 10 for tree cover.")
    p.add_argument("--positive-from-polygons", action="store_true",
                   help="TRUE-LABEL mode: derive the positive mask from the "
                        "--polygons rasterization (pixels inside a field) instead "
                        "of WorldCover. Used for FTW true-field boundaries; needs "
                        "no WorldCover. Negatives = pixels outside any polygon.")
    p.add_argument("--boundary-zone-px", type=int, default=None,
                   help="EDGE CONTROL: restrict sampling (both classes) to "
                        "pixels within K px of a field/non-field boundary, so "
                        "the near-boundary test has both classes. Isolates "
                        "boundary signal from within-field interpolation.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    if args.limit_chips:
        manifest = manifest[: args.limit_chips]
    chip_lookup = {e.chip_id: e for e in manifest}
    chip_ids = list(chip_lookup.keys())

    polygons = gpd.read_parquet(args.polygons)
    polys_by_chip: dict[str, list[dict]] = defaultdict(list)
    for _, p in polygons.iterrows():
        polys_by_chip[p["chip_id"]].append({
            "polygon_id": p["polygon_id"],
            "district": p["district"],
            "geometry": p["geometry"],
            "area_m2": float(p["area_m2"]),
            "size_bin": p["size_bin"],
        })

    banner(f"Per-pixel features: {len(chip_ids)} chips x {len(args.fms)} FMs "
           f"x {args.n_per_class}/class")

    # ----- Pass 1: pre-compute sample indices per chip (chip-seeded RNG so
    # all FMs see the same pixels). Also build polygon-id rasters and sample
    # meta rows.
    log.info("Pass 1: chip-pixel sampling + polygon-id raster build...")
    chip_pixel_idx: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    sample_id = 0
    chips_with_no_label = 0
    chips_without_wc = 0
    for ci, chip_id in enumerate(chip_ids, 1):
        entry = chip_lookup[chip_id]
        polys_here = polys_by_chip.get(chip_id, [])
        poly_id_raster, id_to_meta = build_polygon_id_raster(entry.s2_path, polys_here)

        if args.positive_from_polygons:
            # TRUE-LABEL mode (e.g. FTW): positives = pixels inside a field
            # polygon, negatives = pixels outside any polygon. No WorldCover.
            cropland_mask = (poly_id_raster > 0).astype(np.uint8)
        else:
            if entry.worldcover_path is None or not Path(entry.worldcover_path).exists():
                chips_without_wc += 1
                continue
            with rasterio.open(entry.worldcover_path) as src_wc:
                wc = src_wc.read(1).astype(np.uint8)
            target_code = args.target_class if args.target_class is not None else WORLDCOVER_CROPLAND_CODE
            cropland_mask = (wc == target_code).astype(np.uint8)

        digest = hashlib.sha256(chip_id.encode()).digest()
        chip_seed = args.seed + int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(chip_seed)
        # S2 no-data / all-zero pixels excluded from sampling (both classes).
        with rasterio.open(entry.s2_path) as _src_valid:
            _s2v = _src_valid.read()
        valid_mask = ~np.all(_s2v == 0, axis=0)
        if args.boundary_zone_px:
            from scipy import ndimage
            _fld = cropland_mask.astype(bool)
            _bnd = (_fld & ~ndimage.binary_erosion(_fld)) | (~_fld & ndimage.binary_dilation(_fld))
            _bz = ndimage.binary_dilation(_bnd, iterations=int(args.boundary_zone_px))
            valid_mask = valid_mask & _bz
        rr, cc, ll = sample_pixels_for_chip(cropland_mask, args.n_per_class, rng, valid_mask)
        if len(rr) == 0:
            chips_with_no_label += 1
            continue
        # (poly_id_raster already built above)
        # For each positive sample, look up containing polygon
        sample_pid = poly_id_raster[rr, cc] # 0 if no polygon covers
        # fix: fall back to the chip's TRUE district from the manifest
        # (entry.region.district), NOT chip_id.split('_')[0] which is the
        # longitude string (it polluted the districts list, e.g. "75.51990").
        manifest_district = getattr(getattr(entry, "region", None), "district", None)
        district = (polys_here[0]["district"] if polys_here
                    else (manifest_district or "unknown"))

        for r, c, lbl, pid in zip(rr, cc, ll, sample_pid):
            meta = id_to_meta.get(int(pid)) if pid != 0 else None
            rows.append({
                "sample_id": sample_id,
                "chip_id": chip_id,
                "district": district,
                "label": int(lbl),
                "pixel_r": int(r),
                "pixel_c": int(c),
                "polygon_id": meta["polygon_id"] if meta else None,
                "area_m2": meta["area_m2"] if meta else None,
                "size_bin": meta["size_bin"] if meta else ("non_target" if lbl == 0 else "uncovered_target"),
            })
            sample_id += 1
        chip_pixel_idx[chip_id] = (rr, cc, ll)
        if ci % 50 == 0:
            log.info(" pass1 %d/%d chips samples=%d", ci, len(chip_ids), sample_id)

    log.info(" pass1 done. %d total samples. skipped %d chips (no WC), %d chips (no pixels)",
             sample_id, chips_without_wc, chips_with_no_label)
    if sample_id == 0:
        log.error("no samples collected; aborting")
        return 1

    meta_df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(args.out_dir / "features_per_pixel_meta.parquet")
    log.info("Wrote meta with %d rows", len(meta_df))

    # We need sample positions per chip; index into a contiguous output array
    # via a cumulative counter so all FMs write into the same row positions.
    sample_rows_per_chip: dict[str, np.ndarray] = {}
    chip_id_to_offsets: dict[str, tuple[int, int]] = {}
    cur = 0
    for chip_id in chip_ids:
        if chip_id not in chip_pixel_idx:
            continue
        rr, _, _ = chip_pixel_idx[chip_id]
        n = len(rr)
        chip_id_to_offsets[chip_id] = (cur, cur + n)
        cur += n

    # ----- Pass 2: per FM, run forward on every chip, gather features
    for fm_name in args.fms:
        log.info("FM: %s loading...", fm_name)
        t_load = time.perf_counter()
        fm = get_model(fm_name, device=args.device)
        fm.load()
        log.info(" loaded in %.1fs patch=%s input=%s",
                 time.perf_counter() - t_load, fm.patch_size_px, fm.default_input_size_px)

        feat_dim: int | None = None
        feats: np.ndarray | None = None
        t_fm = time.perf_counter()
        chips_done = 0

        for chip_id in chip_ids:
            if chip_id not in chip_pixel_idx:
                continue
            entry = chip_lookup[chip_id]
            chip, chip_h, chip_w = chip_from_geotiff_array(
                entry.s2_path, scene_date=getattr(entry, "scene_date", None))

            try:
                with torch.no_grad():
                    out = fm.predict(chip, return_tokens=True)
            except Exception as e:
                log.warning("[%s] chip %s forward failed: %s", fm_name, chip_id, e)
                continue
            if out.tokens is None:
                log.warning("[%s] no tokens returned, skipping", fm_name)
                continue

            tokens_1d = out.tokens[0].cpu().numpy()
            # Free the chip's GPU tensors immediately. Without this the CUDA
            # allocator cache balloons (AnySat-dense grew to ~35 GB and OOM'd
            # ~45% of chips on a shared GPU). del + empty_cache caps the
            # footprint to ~one chip + the model. [OOM fix]
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tokens_2d = reshape_to_grid(tokens_1d)
            if tokens_2d is None:
                log.warning("[%s] non-square token count %d", fm_name, tokens_1d.shape[0])
                continue
            h_tok, w_tok, d = tokens_2d.shape

            if feat_dim is None:
                feat_dim = int(d) * (3 if args.multiscale else 1)
                feats = np.zeros((len(meta_df), feat_dim), dtype=np.float32)
                log.info(" feature dim = %d token grid = %dx%d multiscale=%s",
                         feat_dim, h_tok, w_tok, args.multiscale)

            row_idx_map, col_idx_map = build_token_index_maps(
                chip_h, chip_w, h_tok, w_tok,
                fm.default_input_size_px or chip_h,
                fm.patch_size_px or 1,
            )

            rr, cc, _ = chip_pixel_idx[chip_id]
            tr = row_idx_map[rr]
            tc = col_idx_map[cc]
            if args.multiscale:
                pix_feats = _gather_multiscale(tokens_2d, tr, tc, dilations=(0, 1, 3))
            else:
                pix_feats = tokens_2d[tr, tc].astype(np.float32)
            i0, i1 = chip_id_to_offsets[chip_id]
            feats[i0:i1] = pix_feats

            chips_done += 1
            if chips_done % 20 == 0:
                rate = chips_done / max(time.perf_counter() - t_fm, 1e-6)
                eta = (len(chip_pixel_idx) - chips_done) / max(rate, 1e-6)
                log.info(" [%s] %d/%d chips (%.2f chips/s eta %.0fs)",
                         fm_name, chips_done, len(chip_pixel_idx), rate, eta)

        if feats is None:
            log.error("[%s] no chips processed", fm_name)
        else:
            out_path = args.out_dir / f"features_{fm_name.replace('-', '_').replace('.', '_')}.npy"
            np.save(out_path, feats)
            log.info("[%s] saved %s shape=%s elapsed=%.0fs",
                     fm_name, out_path, feats.shape, time.perf_counter() - t_fm)

        del fm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    banner("Per-pixel extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
