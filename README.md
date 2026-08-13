# Auditing GeoFM Evaluation for Field-Extent Segmentation

Code and results for the TMLR paper **"Auditing GeoFM Evaluation for
Field-Extent Segmentation: Label Proxies, Baselines, and When Frozen Features
Match Fine-tuning."**

- Paper and review record: <https://openreview.net/forum?id=qRXVTe1yYp>
- Repository: <https://github.com/thesantoshpant/auditing-geofm-evaluation>

## What the study tests

The study examines agricultural field-extent segmentation from single-date
Sentinel-2 imagery in six countries. It first measures how two evaluation
choices change the apparent value of geospatial foundation models (GeoFMs):

1. replacing field-polygon labels with ESA WorldCover cropland labels; and
2. comparing GeoFMs only with a per-pixel spectral baseline rather than a
   trained segmentation model.

It then compares a frozen GeoFM backbone with a trained decoder against full
fine-tuning. The comparison uses two GeoFMs, ten matched seeds, the same
decoder, and the same 80-epoch schedule.

## Main results

- Label and baseline choice both change the apparent GeoFM advantage on the
  same evaluation pixels.
- Frozen-decoder and full fine-tuning are TOST-equivalent within 0.02 AUROC in
  9 of 12 in-region region-by-model cells. Kenya is the consistent
  sparse-positive exception; TerraMind India is non-equivalent in the
  frozen-favored direction.
- A from-scratch U-Net has the highest mean AUROC on the confirmatory full
  matrix of 30 directed transfers. The non-Kenya 20-transfer analysis is an
  exploratory diagnostic: there, both frozen GeoFMs outperform their
  full-fine-tuned counterparts, and TerraMind frozen is close to the U-Net.
- The Prithvi frozen decoder trains 2.75 million parameters, compared with
  about 307 million for full Prithvi fine-tuning. The TerraMind decoder has
  2.16 million trainable parameters. Frozen and fully fine-tuned variants execute the same
  backbone-plus-decoder graph at inference.

These claims concern single-date, per-pixel field-extent segmentation. They do
not establish parcel-boundary or instance-delineation performance.

## Configuration key

| Name | Trainable component | Purpose |
|---|---|---|
| frozen probe | linear head on fixed features | label and baseline audit |
| frozen decoder | decoder on a fixed backbone (2.75M Prithvi; 2.16M TerraMind) | primary adaptation comparison |
| full fine-tuning | backbone and the same decoder | primary adaptation comparison |
| U-Net | 31M-parameter model trained from scratch | supervised baseline |

The command `ftw_finetune_fm.py --freeze backbone` trains the **frozen
decoder**, not the linear probe. Linear probes are produced by the controlled
label-comparison pipeline.

## Repository layout

```text
paper/              TMLR source, references, figures, and compiled paper
scripts/            Data preparation, training, aggregation, and checks
src/ftw_eval/       Importable model and data utilities
data/results/       Raw run outputs and derived, checksummed results
data/index/         Public-data indices
docs/               Detailed reproduction and evaluation notes
tests/              Unit tests
ARTIFACT.md         Artifact map and reproduction instructions
croissant.json      Machine-readable dataset and artifact metadata
```

Raw Sentinel-2 imagery, WorldCover rasters, and Fields of The World polygons
are not redistributed. The scripts retrieve them from their public sources.

## Verify the released results

The exact top-level versions captured for the headline training grid are in
`requirements.runtime.txt` and `data/results/environment.json`. Install that
environment with Python 3.11 and CUDA 12.1 using:

```bash
pip install -r requirements.runtime.txt
pip install -e .
```

`requirements.lock` is a separately resolved current environment for the
CPU-side analysis and reproduction scripts:

```bash
pip install -r requirements.lock
pip install -e .
python scripts/checksums.py --verify
```

To regenerate every derived camera-ready summary from the released raw JSONs
and then check the release manifest:

```bash
bash scripts/reproduce_all.sh
```

The command does not launch GPU training. It rebuilds the headline analysis,
cross-region metrics, equivalence-margin sensitivity, epoch control, regional
diagnostic, convergence figure, and multiple-comparison summary.

## Reproduce a country pipeline

Google Earth Engine authentication is required for the imagery pull.

```bash
C=india
bash scripts/prep_ftw_country.sh "$C"
bash scripts/run_ftw_country.sh "$C"
python scripts/ftw_export_split.py --country "$C"
python scripts/ftw_controlled_label_comparison.py --country "$C"
python scripts/ftw_unet_baseline.py --robust "$C"
python scripts/ftw_finetune_fm.py --model prithvi --freeze backbone "$C"
python scripts/ftw_finetune_fm.py --model prithvi "$C"
```

See [ARTIFACT.md](ARTIFACT.md) and
[docs/FTW_REPRODUCE.md](docs/FTW_REPRODUCE.md) for the full workflow,
dependency caveats, and per-model instructions.

## Data and licenses

- Fields of The World: CC BY 4.0
- ESA WorldCover 10 m v200: CC BY 4.0
- Sentinel-2 L2A: Copernicus free and open data
- Repository code: Apache License 2.0
- Derived metrics and documentation: CC BY 4.0

## Citation

```bibtex
@article{pant2026auditing,
  title   = {Auditing GeoFM Evaluation for Field-Extent Segmentation:
             Label Proxies, Baselines, and When Frozen Features Match Fine-tuning},
  author  = {Pant, Santosh},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=qRXVTe1yYp}
}
```
