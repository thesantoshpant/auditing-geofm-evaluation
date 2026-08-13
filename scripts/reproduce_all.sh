#!/usr/bin/env bash
# Re-aggregate and verify the released headline artifacts.
# Full model/data reproduction is documented in ARTIFACT.md; this script checks
# the shipped release state without launching new GPU training jobs.
set -euo pipefail

if [[ -z "${PY:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo "No Python interpreter found. Set PY to the Python 3.11 executable." >&2
    exit 2
  fi
fi

$PY scripts/integrate_headline_results.py
$PY scripts/integrate_transfer_metrics.py
$PY scripts/ftw_unet_baseline_epochs.py
$PY scripts/integrate_revision_controls.py
$PY scripts/ftw_finetune_fm_curves.py
$PY scripts/ftw_multiple_comparison.py
$PY scripts/checksums.py --verify

echo "Release verification complete."
echo "Headline aggregates:"
echo "  data/results/ftw_inregion_equivalence.json"
echo "  data/results/ftw_cross_region_transfer.json"
echo "  data/results/ftw_headline_summary.txt"
echo "  data/results/ftw_xfer_metrics.json"
echo "  data/results/ftw_unet_epoch_sensitivity.json"
echo "  data/results/ftw_eps_sweep.json"
echo "  data/results/ftw_regional_regime.json"
echo "  data/results/ftw_convergence_diagnostic.json"
