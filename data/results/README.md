# Result File Map

This directory contains released run-level metrics and deterministic aggregates.
Raw imagery, label polygons, model weights, and feature tensors are rebuilt from
their public sources and are not redistributed. `CHECKSUMS.sha256` covers every
canonical file used by the paper.

## Headline grid and aggregates

| Pattern or file | Meaning | Generator |
|---|---|---|
| `ftw_finetune_fm_{prithvi,terramind}_<region>_{frozen_decoder,full_finetune}_seed<0-9>.json` | 240 GeoFM runs; each stores in-region and five transfer evaluations | `scripts/ftw_finetune_fm.py --eval-all-regions` |
| `ftw_unet_<region>_seed<0-9>.json` | 60 U-Net runs; each stores in-region and five transfer evaluations | `scripts/ftw_unet_baseline.py --eval-all-regions` |
| `ftw_inregion_equivalence.json` | 12-cell seed-paired TOST analysis | `scripts/integrate_headline_results.py` |
| `ftw_cross_region_transfer.json` | full 30-transfer AUROC matrix and paired comparisons | `scripts/integrate_headline_results.py` |
| `ftw_headline_summary.txt` | readable headline summary | `scripts/integrate_headline_results.py` |
| `ftw_xfer_metrics.json` | transfer AUROC, average precision, and IoU | `scripts/integrate_transfer_metrics.py` |
| `ftw_param_counts.json` | trainable-parameter counts and Prithvi ratio | `scripts/report_param_counts.py` |

## Revision and sensitivity controls

| Pattern or file | Meaning | Generator |
|---|---|---|
| `ftw_eps_sweep.json` | equivalence-margin sensitivity | `scripts/integrate_revision_controls.py` |
| `ftw_regional_regime.json` | regional descriptors, mean polygon areas derived from the alignment summary, and Cambodia low-data control | `scripts/integrate_revision_controls.py` |
| `ftw_unet_epoch_sensitivity.json` | 80- versus 150-epoch U-Net control | `scripts/ftw_unet_baseline_epochs.py` |
| `ftw_convergence_diagnostic.json` | 150-epoch Prithvi trajectories | `scripts/ftw_finetune_fm_curves.py` |
| `ftw_multiple_comparison.json` | corrected headline directional tests | `scripts/ftw_multiple_comparison.py` |
| `ftw_controlled_label_comparison_<region>*.json` | same-pixel polygon-label versus WorldCover comparison | `scripts/ftw_controlled_label_comparison.py` |
| `ftw_bootstrap_ci_<region>.json` | chip-clustered confidence intervals | `scripts/ftw_bootstrap_ci.py` |
| `ftw_split_<region>.json` | canonical chip-grouped split metadata | `scripts/ftw_export_split.py` |
| `ftw_proxy_sensitivity_<region>.json` | retained field-size sensitivity summaries | `scripts/ftw_field_sizes.py` |
| `ftw_edge_zone_eval_india.json` | frozen-probe boundary-zone control | `scripts/ftw_edge_eval.py` |
| `ftw_partial_label_sensitivity.json` | high-confidence-negative control | `scripts/ftw_partial_label_sensitivity.py` |
| `ftw_india_wilcoxon.json` | one-off 12-chunk India Prithvi frozen-probe versus U-Net directional audit | aggregate retained and checksummed; per-chunk prediction arrays were not retained |

## Source runs for revision controls

| Pattern or file | Meaning | Producer or consumer |
|---|---|---|
| `ftw_unet_e80_seed*.json` | three-seed, 80-epoch U-Net budget control | produced by `scripts/ftw_unet_baseline.py`; summarized by `scripts/ftw_unet_baseline_epochs.py` |
| `ftw_finetune_fm_prithvi_curve150_*.json` | single-seed 150-epoch Prithvi convergence trajectories | produced by `scripts/ftw_finetune_fm.py`; summarized by `scripts/ftw_finetune_fm_curves.py` |
| `ftw_{unet,finetune_fm_prithvi}_camld_f*_s*.json` | Cambodia chip-count subsampling control | produced by the corresponding U-Net or GeoFM trainer; summarized by `scripts/integrate_revision_controls.py` |
| `ftw_{unet,finetune_fm_prithvi}_{ld,ldFN}_f*_s*.json` | retained exploratory low-data runs for the other regions | corresponding U-Net or GeoFM trainer; not used in a headline aggregate |
| `ftw_{unet,finetune_fm_*}_{probe*,seedB*,seedD*,camB*}.json` | earlier three-seed in-region recipe runs retained for traceability | corresponding U-Net or GeoFM trainer; superseded by the ten-seed grid |
| `ftw_{unet,finetune_fm_prithvi}_xr_*.json` | earlier curated-transfer pilot runs | corresponding U-Net or GeoFM trainer; superseded by the full 30-transfer grid |
| `ftw_finetune_fm_prithvi_xm_*.json`, `ftw_unet_xmetrics.json` | single-run threshold-metric diagnostics | corresponding trainer; superseded by ten-seed metric aggregates |
| `ftw_finetune_fm_{prithvi,terramind}.json`, `ftw_unet_{baseline,baseline_robust}.json`, `ftw_{unet,finetune_fm_*}_seed*.json` | pre-grid pilot summaries and runs | corresponding trainer; retained for provenance, not used by the camera-ready paper |
| `ftw_alignment_summary.json`, `ftw_transfer_probe_multi.json` | frozen-probe alignment and transfer diagnostics | `scripts/ftw_alignment_check.py` or `scripts/ftw_transfer_probe.py` |
| `ftw_{probes_summary,transfer_probe,transfer_summary}.json` | early frozen-probe summaries retained for provenance | legacy aggregate outputs; not used by the camera-ready paper |
| `ftw_{edge_pixel_ablation,unet_perpixel_ablation,windowed_rf_control}.json` | spatial-context ablations | `scripts/ftw_edge_ablation.py`, `scripts/ftw_unet_baseline.py --perpixel`, or `scripts/ftw_windowed_rf_control.py` |

These older source runs remain at the result root because some revision controls
refer to them directly. They are checksummed, but they are not mixed into the
single-recipe ten-seed headline analysis. Superseded aggregate logic is kept
under `scripts/archive/historical_iterations/`, and the stale pilot master
summary is kept under `archive/historical_iterations/`.

`environment.json` records the software environment used for the headline
training grid. Files under `archive/historical_iterations/` are retained only
for provenance and are not used by the paper or canonical checksum manifest.

Run `bash scripts/reproduce_all.sh` from any working directory to regenerate
the derived files and verify the release manifest. Exact commands for rebuilding
the 300-file training grid are in `ARTIFACT.md`.
