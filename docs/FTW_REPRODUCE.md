# Reproducing the FTW Label and Frozen-Probe Controls

This guide covers the label-source audit, frozen linear probes, and related
spatial controls for all six regions. Trained U-Net, frozen-decoder,
full-fine-tuning, response experiments, and release-level verification are
documented in `ARTIFACT.md`.

Large chip rasters and feature arrays are not included in the release. They are
regenerated from Fields of The World (FTW), Sentinel-2, and ESA WorldCover.
Run-level and derived result JSONs are included and checksummed.

## Environment

Use Python 3.11 and install the dependencies described in `ARTIFACT.md`:

```bash
pip install -r requirements.lock
pip install -e ".[all]"
```

Google Earth Engine authentication is required for Sentinel-2 and WorldCover
retrieval. FTW downloads may require a browser-style user-agent at the source.

## Per-region pipeline

Choose one of `india`, `cambodia`, `vietnam`, `kenya`, `france`, or
`netherlands`.

```bash
C=vietnam

# FTW labels and keyed polygons
bash scripts/prep_ftw_country.sh "$C"

# Public-data index, imagery pull, and feature extraction
python scripts/build_ftw_index.py --country "$C" --limit 800 \
  --out "data/index/ftw_${C}.jsonl"
bash scripts/run_ftw_country.sh "$C"

# Canonical chip-grouped split and same-pixel label comparison
python scripts/ftw_export_split.py --country "$C"
python scripts/ftw_controlled_label_comparison.py --country "$C"

# Tile-disjoint sensitivity
python scripts/ftw_controlled_label_comparison.py --country "$C" --group-by tile
```

## Design checks

- `ftw_controlled_label_comparison.py` builds one shared pixel table and scores
  it against polygon-derived field membership and WorldCover cropland labels.
- Every split is grouped by chip. Result files record zero chip overlap.
- Spectral and frozen-probe models use the same evaluation rows, masks, and
  split.
- `ftw_alignment_check.py` verifies polygon-to-raster alignment directly after
  projection to each chip's grid.
- The tile-disjoint sensitivity is informative only where enough MGRS tiles are
  available; its uncertainty should not be inferred from the chip-grouped run.

## Principal files

- `ftw_controlled_label_comparison*.json`: same-pixel label audit
- `ftw_bootstrap_ci_<region>.json`: chip-clustered AUROC intervals and paired
  deltas for five regions; Cambodia's retained per-chip features are unavailable
- `ftw_edge_zone_eval_india.json`: boundary-zone results for five frozen-probe
  configurations only
- `ftw_partial_label_sensitivity.json`: high-confidence-negative sensitivity in
  India, France, and Kenya
- `ftw_proxy_mismatch_coverage.json`: polygon-field and WorldCover coverage
- `eval_per_pixel_ftw_*_chip.json`: standalone probe and spectral evaluations

Verify all released files with:

```bash
python scripts/checksums.py --verify
```

The release does not host regenerated chip imagery, feature arrays, or FTW label
parquets. Those files can be rebuilt from their public sources with the commands
above.
