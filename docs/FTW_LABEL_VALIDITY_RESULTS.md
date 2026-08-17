# Label and Baseline Audit Results

This note summarizes the controlled audit reported in the paper. The task is
per-pixel membership inside an FTW field polygon. ESA WorldCover cropland is a
different, more spectrally separable land-cover target.

## Same-pixel comparison

The random forest is evaluated on the same pixels under both labels. The U-Net,
Prithvi frozen probe, and fine-tuned Prithvi columns use polygon-derived
field-extent labels.

| Region | RF field label | RF WorldCover proxy | U-Net | Prithvi frozen probe | Fine-tuned Prithvi |
|---|---:|---:|---:|---:|---:|
| India | 0.574 | 0.786 | 0.941 | 0.983 | 0.985 |
| Cambodia | 0.550 | 0.900 | 0.948 | 0.924 | 0.949 |
| Vietnam | 0.643 | 0.911 | 0.964 | 0.929 | 0.958 |
| Kenya | 0.651 | 0.806 | 0.891 | 0.766 | 0.771 |
| France | 0.695 | 0.959 | 0.977 | 0.961 | 0.979 |
| Netherlands | 0.815 | 0.928 | 0.981 | 0.955 | 0.958 |

Values are AUROC. The proxy raises random-forest AUROC in every region, by
0.11 to 0.35. This does not show that one label is a noisier measurement of the
other; it shows that they define different tasks.

On field-extent labels, the U-Net has higher point AUROC than the Prithvi frozen
linear probe in five of six regions. India is the exception. The chip-paired
India comparison gives Wilcoxon `p = 2.4e-4`. The paper reports ten-seed U-Net
and fine-tuning intervals and chip-clustered frozen-probe intervals where the
retained per-chip data permit them.

## Scope of the controls

- A 5 by 5 windowed random forest improves mean AUROC by about 0.01, so local
  pooling does not close the gap to the spatial models.
- The India boundary-zone analysis evaluates four frozen probes plus the
  per-pixel random forest. Their AUROCs are near 0.58 on that slice. It does not
  evaluate the U-Net, frozen decoder, or full fine-tuning and therefore cannot
  rank those trained segmentation models at field edges.
- Restricting negatives to high-confidence non-cropland pixels was tested in
  India, France, and Kenya. Prithvi changes little, while the random forest can
  change substantially, especially in France. This is a scoped sensitivity
  result, not a claim that every conclusion is invariant.
- Tile-disjoint results are reported per model. The small number of tiles in
  some regions makes those estimates less stable than the chip-grouped results.

## Provenance

The controlled values are in
`data/results/ftw_controlled_label_comparison*.json`; trained-model values are
integrated from the ten-seed run files. Relevant uncertainty and sensitivity
files include `ftw_bootstrap_ci_*.json`, `ftw_india_wilcoxon.json`,
`ftw_windowed_rf_control.json`, `ftw_edge_zone_eval_india.json`, and
`ftw_partial_label_sensitivity.json`.

Run `python scripts/checksums.py --verify` to verify the released artifacts.
