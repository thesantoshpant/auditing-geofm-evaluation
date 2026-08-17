#!/usr/bin/env bash
# Retrieve public imagery and run the frozen-probe controls for one country.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_ftw_country.sh <country> [output-dir]" >&2
  exit 2
fi
C="$1"
OUTDIR="${2:-data/results}"
case "$C" in
  india|cambodia|vietnam|kenya|france|netherlands) ;;
  *) echo "Unsupported country: $C" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PY:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo "No Python interpreter found. Set PY to the project interpreter." >&2
    exit 2
  fi
fi

IDX="data/index/ftw_${C}.jsonl"
CH="data/chips_ftw_${C}"
M="data/chips/manifest_ftw_${C}.jsonl"
PKEY="data/labels/polygons_ftw_${C}_keyed.parquet"
mkdir -p data/index data/chips data/labels "$CH" "$OUTDIR"
if [[ ! -f "$IDX" ]]; then
  echo "$IDX is missing. Run scripts/build_ftw_index.py first." >&2
  exit 2
fi

echo "### [1] PULL S2 ($C)"
"$PY" scripts/pull_chip_imagery.py --index "$IDX" --out-dir "$CH" --skip-s1 --workers 8
echo "### [2] PULL WC"
"$PY" scripts/pull_worldcover_labels.py --index "$IDX" --out-dir "$CH" --workers 8
echo "### [3] MANIFEST"
"$PY" scripts/build_chip_manifest.py --index "$IDX" --chip-dir "$CH" --out "$M" --require-worldcover
echo "### [4] REKEY"
C="$C" "$PY" - <<'PYEOF'
import json
import os

import geopandas as gpd

country = os.environ["C"]
with open(f"data/index/ftw_{country}.jsonl", encoding="utf-8") as handle:
    index = {
        row["aoi_id"]: "%.5f_%.5f_%d" % (row["lon"], row["lat"], int(row["scene_date"]) // 1000)
        for row in (json.loads(line) for line in handle)
    }
with open(f"data/chips/manifest_ftw_{country}.jsonl", encoding="utf-8") as handle:
    manifest = {json.loads(line)["chip_id"] for line in handle}
polygons = gpd.read_parquet(f"data/labels/polygons_ftw_{country}.parquet")
polygons["chip_id"] = polygons["chip_id"].map(index)
polygons = polygons[polygons["chip_id"].notna() & polygons["chip_id"].isin(manifest)].copy()
polygons["polygon_id"] = [f"{chip}_{i}" for i, chip in enumerate(polygons["chip_id"])]
polygons.to_parquet(f"data/labels/polygons_ftw_{country}_keyed.parquet")
print(f"rekey {country}: {len(polygons)} polygons, {polygons['chip_id'].nunique()} chips")
PYEOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "### [5] TRUE 3FM"
"$PY" scripts/extract_features_per_pixel.py --manifest "$M" --polygons "$PKEY" --out-dir "data/features_per_pixel_ftw_${C}_true" --positive-from-polygons
echo "### [6] TRUE AnySat"
"$PY" scripts/extract_features_per_pixel.py --manifest "$M" --polygons "$PKEY" --out-dir "data/features_per_pixel_ftw_${C}_true_anysat" --positive-from-polygons --fms anysat-dense
echo "### [7] PROXY 3FM"
"$PY" scripts/extract_features_per_pixel.py --manifest "$M" --polygons "$PKEY" --out-dir "data/features_per_pixel_ftw_${C}_proxy"
echo "### [8] PROXY AnySat"
"$PY" scripts/extract_features_per_pixel.py --manifest "$M" --polygons "$PKEY" --out-dir "data/features_per_pixel_ftw_${C}_proxy_anysat" --fms anysat-dense
echo "### [9] CONTROLLED"
"$PY" scripts/ftw_controlled_label_comparison.py --country "$C" \
  --output "$OUTDIR/ftw_controlled_label_comparison_${C}.json"
echo "### [10] STANDALONE"
for label in true proxy; do
  "$PY" scripts/eval_per_pixel.py --features-dir "data/features_per_pixel_ftw_${C}_${label}" --out "$OUTDIR/eval_per_pixel_ftw_${C}_${label}_chip.json" --split chip --seed 20260514
  "$PY" scripts/eval_per_pixel.py --features-dir "data/features_per_pixel_ftw_${C}_${label}_anysat" --out "$OUTDIR/eval_per_pixel_ftw_${C}_${label}_anysat_chip.json" --split chip --seed 20260514
  "$PY" scripts/eval_nonfm_baseline_per_pixel.py --manifest "$M" --features-dir "data/features_per_pixel_ftw_${C}_${label}" --out "$OUTDIR/eval_per_pixel_ftw_${C}_${label}_nonfm_rf_chip.json" --clf rf --split chip --seed 20260514
done
echo "### [11] BOOTSTRAP"
"$PY" scripts/ftw_bootstrap_ci.py --country "$C" \
  --output "$OUTDIR/ftw_bootstrap_ci_${C}.json"
echo "### $C PIPELINE DONE"
