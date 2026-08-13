# Evaluation Protocol for GeoFM Field-Extent Segmentation

This checklist records the controls used in the TMLR paper. The task is
per-pixel field-extent membership from single-date Sentinel-2 imagery. It is not
parcel-boundary or instance delineation.

## Labels

1. **Use labels that match the task.** Field-polygon membership and land-cover
   cropland are different targets. If a proxy is unavoidable, name the proxy
   task and treat rankings as provisional.
2. **Hold evaluation pixels fixed when comparing labels.** Score one shared
   pixel set under both label definitions. Report the pixel count and verify
   that no chip appears in both train and test sets.

## Splits and preprocessing

3. **Group the split by chip.** Pixel-level random splits leak local spatial
   structure. Add a tile-disjoint sensitivity check when the available number
   of tiles makes it informative.
4. **Match each model's documented input pipeline.** Record band order,
   reflectance scaling, normalization, spatial resolution, and no-data masks.
5. **Verify label-to-image alignment directly.** Use coordinate and raster
   checks rather than treating a high model score as evidence of alignment.

## Baselines and adaptation

6. **Include both spectral and spatial supervised baselines.** A per-pixel
   spectral model tests whether the label is spectrally separable. A trained
   segmentation model such as U-Net tests whether a GeoFM improves on a strong
   supervised model.
7. **Separate probe, decoder, and fine-tuning results.** State which parameters
   are trainable, use the same decoder for frozen-backbone and full-fine-tuning
   comparisons, and report trainable parameter counts.
8. **Audit training-budget differences.** Match budgets for the central
   adaptation comparison. When architecture-specific schedules differ, report
   a budget sensitivity or convergence diagnostic and describe its scope.

## Metrics and uncertainty

9. **Report ranking and threshold metrics.** AUROC and average precision assess
   ranking, while IoU or F1 at a declared threshold assesses the deployed mask.
   Sparse-positive regions can look different under these metric families.
10. **Match uncertainty to the sampling unit.** Use chip-clustered intervals for
    chip-level comparisons and seed-paired intervals for matched training runs.
    Declare equivalence margins before interpreting TOST results, show margin
    sensitivity, and state the family used for multiplicity correction.

## Interpretation

- Treat the full transfer matrix as the primary cross-region analysis. Label
  any subset defined after observing a failure case as exploratory.
- Do not infer boundary understanding from a field-extent score. The released
  boundary-zone control covers frozen probes only and cannot rank U-Net,
  frozen-decoder, or full-fine-tuned segmentation models on edge pixels.
- Describe regional mechanisms as hypotheses unless they are isolated by a
  controlled experiment. The six-region sparsity analysis in this artifact is
  descriptive and does not estimate a causal threshold.

All released metric files are protected by
`data/results/CHECKSUMS.sha256`. End-to-end commands and known limits are in
`ARTIFACT.md` and `docs/FTW_REPRODUCE.md`.
