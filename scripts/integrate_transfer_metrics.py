#!/usr/bin/env python
"""Aggregate cross-region AUROC, AP, and IoU from the 300-run headline grid."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from integrate_headline_results import REGIONS, SEEDS, get_value, load_all


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "results" / "ftw_xfer_metrics.json"
CONFIGS = {
    "unet": "unet",
    "prithvi_frozen": "prithvi:backbone",
    "prithvi_fullft": "prithvi:none",
    "terramind_frozen": "terramind:backbone",
    "terramind_fullft": "terramind:none",
}


def main() -> int:
    fm, unet = load_all()
    all_pairs = [(a, b) for a in REGIONS for b in REGIONS if a != b]
    subsets = {
        "all30": all_pairs,
        "nonkenya20": [
            (a, b) for a, b in all_pairs if a != "kenya" and b != "kenya"
        ],
    }
    output: dict[str, object] = {
        "method": (
            "For each directed train-region to test-region pair and configuration, "
            "metrics are averaged over the same ten seeds. Subset summaries then "
            "average the 30 or 20 pair means."
        ),
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "subsets": {},
        "per_pair_per_config": {},
    }

    for config_name, config_label in CONFIGS.items():
        output["per_pair_per_config"][config_name] = {}
        for source, target in all_pairs:
            pair = f"{source}->{target}"
            output["per_pair_per_config"][config_name][pair] = {}
            for metric in ("auroc", "ap", "iou"):
                values = [
                    get_value(
                        config_label,
                        source,
                        target,
                        seed,
                        fm,
                        unet,
                        metric=metric,
                    )
                    for seed in SEEDS
                ]
                output["per_pair_per_config"][config_name][pair][metric] = {
                    "mean": statistics.fmean(values),
                    "seed_values": values,
                }

    for config_name in CONFIGS:
        output["subsets"][config_name] = {}
        for subset_name, pairs in subsets.items():
            summary: dict[str, object] = {"n_pairs": len(pairs)}
            for metric in ("auroc", "ap", "iou"):
                pair_means = [
                    output["per_pair_per_config"][config_name][f"{a}->{b}"][metric][
                        "mean"
                    ]
                    for a, b in pairs
                ]
                summary[metric] = round(statistics.fmean(pair_means), 4)
            output["subsets"][config_name][subset_name] = summary

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
