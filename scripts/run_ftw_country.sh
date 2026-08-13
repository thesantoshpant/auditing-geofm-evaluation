C="$1"
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
IDX=data/index/ftw_${C}.jsonl; CH=data/chips_ftw_${C}
M=data/chips/manifest_ftw_${C}.jsonl; PKEY=data/labels/polygons_ftw_${C}_keyed.parquet
echo "### [1] PULL S2 ($C)"; $PY scripts/pull_chip_imagery.py --index $IDX --out-dir $CH --skip-s1 --workers 8 || echo "(s2 nonzero; continuing)"
echo "### [2] PULL WC";       $PY scripts/pull_worldcover_labels.py --index $IDX --out-dir $CH --workers 8 || echo "(wc nonzero; continuing)"
echo "### [3] MANIFEST";      $PY scripts/build_chip_manifest.py --index $IDX --chip-dir $CH --out $M --require-worldcover
echo "### [4] REKEY";         C="$C" $PY - <<'PYEOF'
import os, json, geopandas as gpd
C=os.environ["C"]
idx={}
for ln in open(f"data/index/ftw_{C}.jsonl"):
    r=json.loads(ln); idx[r["aoi_id"]]="%.5f_%.5f_%d"%(r["lon"],r["lat"],int(r["scene_date"])//1000)
man=set(json.loads(l)["chip_id"] for l in open(f"data/chips/manifest_ftw_{C}.jsonl"))
g=gpd.read_parquet(f"data/labels/polygons_ftw_{C}.parquet")
g["chip_id"]=g["chip_id"].map(idx); g=g[g["chip_id"].notna() & g["chip_id"].isin(man)].copy()
g["polygon_id"]=[f"{c}_{i}" for i,c in enumerate(g["chip_id"])]
g.to_parquet(f"data/labels/polygons_ftw_{C}_keyed.parquet")
print("rekey %s: %d polygons, %d chips overlap manifest"%(C,len(g),g["chip_id"].nunique()))
PYEOF
export CUDA_VISIBLE_DEVICES=0
echo "### [5] TRUE 3FM";   $PY scripts/extract_features_per_pixel.py --manifest $M --polygons $PKEY --out-dir data/features_per_pixel_ftw_${C}_true --positive-from-polygons 2>&1 | tail -2
echo "### [6] TRUE anysat"; $PY scripts/extract_features_per_pixel.py --manifest $M --polygons $PKEY --out-dir data/features_per_pixel_ftw_${C}_true_anysat --positive-from-polygons --fms anysat-dense 2>&1 | tail -2
echo "### [7] PROXY 3FM";  $PY scripts/extract_features_per_pixel.py --manifest $M --polygons $PKEY --out-dir data/features_per_pixel_ftw_${C}_proxy 2>&1 | tail -2
echo "### [8] PROXY anysat"; $PY scripts/extract_features_per_pixel.py --manifest $M --polygons $PKEY --out-dir data/features_per_pixel_ftw_${C}_proxy_anysat --fms anysat-dense 2>&1 | tail -2
echo "### [9] CONTROLLED";  $PY scripts/ftw_controlled_label_comparison.py --country $C 2>&1 | grep -vE "Convergence|n_iter|STOP|Increase|scikit|refer"
echo "### [10] STANDALONE"
for L in true proxy; do
  $PY scripts/eval_per_pixel.py --features-dir data/features_per_pixel_ftw_${C}_$L --out data/results/eval_per_pixel_ftw_${C}_${L}_chip.json --split chip --seed 20260514 2>&1 | grep -E "F1="
  $PY scripts/eval_per_pixel.py --features-dir data/features_per_pixel_ftw_${C}_${L}_anysat --out data/results/eval_per_pixel_ftw_${C}_${L}_anysat_chip.json --split chip --seed 20260514 2>&1 | grep -E "F1="
  $PY scripts/eval_nonfm_baseline_per_pixel.py --manifest $M --features-dir data/features_per_pixel_ftw_${C}_$L --out data/results/eval_per_pixel_ftw_${C}_${L}_nonfm_rf_chip.json --clf rf --split chip --seed 20260514 2>&1 | grep -E "F1="
done
echo "### [11] BOOTSTRAP"; $PY scripts/ftw_bootstrap_ci.py --country $C 2>&1 | grep -vE "Convergence|n_iter|STOP|Increase|scikit|refer"
echo "### $C PIPELINE DONE"
