#!/usr/bin/env python
"""Regenerate the equivalence-margin and regional-regime control artifacts."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
INREGION = RESULTS / "ftw_inregion_equivalence.json"
EPS_OUTPUT = RESULTS / "ftw_eps_sweep.json"
REGIME_OUTPUT = RESULTS / "ftw_regional_regime.json"
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]
MODELS = ["prithvi", "terramind"]
EPSILONS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# Rounded training-mask prevalence reported in the manuscript. The source masks
# are not redistributed; all other fields in the regime table are recomputed
# below from shipped JSONs.
POSITIVE_PIXEL_PCT = {
    "india": 0.3,
    "cambodia": 23.0,
    "vietnam": 18.0,
    "kenya": 0.1,
    "france": 11.0,
    "netherlands": 3.8,
}
REGIME_LABEL = {
    "india": "smallholder; model-dependent",
    "cambodia": "tie",
    "vietnam": "tie",
    "kenya": "sparse-positive collapse",
    "france": "industrial; near-parity",
    "netherlands": "frozen-favored AUROC",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    center = mean(values)
    half_width = float(stats.t.ppf(0.975, len(values) - 1)) * sample_std(
        values
    ) / math.sqrt(len(values))
    return center - half_width, center + half_width


def tost_pvalue(delta_mean: float, delta_std: float, n: int, epsilon: float) -> float:
    if delta_std == 0:
        return 0.0 if abs(delta_mean) < epsilon else 1.0
    standard_error = delta_std / math.sqrt(n)
    lower_t = (delta_mean + epsilon) / standard_error
    upper_t = (epsilon - delta_mean) / standard_error
    # Direct survival probabilities remain stable in the small tails where
    # ``1 - cdf`` suffers catastrophic cancellation.
    lower_p = float(stats.t.sf(lower_t, n - 1))
    upper_p = float(stats.t.sf(upper_t, n - 1))
    return max(lower_p, upper_p)


def canonical_pvalue(value: float) -> float:
    """Serialize p-values at stable precision across supported SciPy releases."""
    return float(f"{value:.12g}")


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, min(1.0, (len(p_values) - rank) * p_value))
        adjusted[index] = running
    return adjusted


def build_epsilon_sweep(inregion: dict) -> dict:
    cells: list[dict] = []
    for model in MODELS:
        for region in REGIONS:
            source = inregion["models"][model]["regions"][region]
            deltas = [
                float(frozen) - float(full_ft)
                for frozen, full_ft in zip(
                    source["frozen_seeds"], source["fullft_seeds"], strict=True
                )
            ]
            lower, upper = ci95(deltas)
            cells.append(
                {
                    "model": model,
                    "region": region,
                    "paired_delta_mean": mean(deltas),
                    "paired_delta_std": sample_std(deltas),
                    "ci95": [lower, upper],
                }
            )

    sweep: dict[str, object] = {
        "method": (
            "Sensitivity of the primary equivalence verdict to epsilon. The table "
            "criterion is containment of the 95% seed-paired t-interval within "
            "+/-epsilon. TOST p-values and Holm adjustments are also provided for "
            "audit, but the paper's sensitivity counts use unadjusted CI containment. "
            "P-values are serialized to 12 significant digits."
        ),
        "n_seeds": 10,
        "df": 9,
        "epsilons": {},
    }
    for epsilon in EPSILONS:
        p_values = [
            canonical_pvalue(tost_pvalue(
                cell["paired_delta_mean"], cell["paired_delta_std"], 10, epsilon
            ))
            for cell in cells
        ]
        adjusted = [canonical_pvalue(value) for value in holm_adjust(p_values)]
        rows = []
        for cell, p_value, adjusted_p in zip(cells, p_values, adjusted, strict=True):
            lower, upper = cell["ci95"]
            equivalent = lower >= -epsilon and upper <= epsilon
            rows.append(
                {
                    "model": cell["model"],
                    "region": cell["region"],
                    "paired_delta_mean": round(cell["paired_delta_mean"], 6),
                    "ci95": [round(lower, 6), round(upper, 6)],
                    "direction": (
                        "frozen-favored"
                        if cell["paired_delta_mean"] > 0
                        else "full-ft-favored"
                    ),
                    "equivalent_by_ci95": equivalent,
                    "tost_p_value": p_value,
                    "tost_p_value_holm": adjusted_p,
                    "retained_after_holm_0_05": adjusted_p < 0.05,
                }
            )
        sweep["epsilons"][f"{epsilon:.3f}"] = {
            "n_equivalent_by_ci95": sum(
                row["equivalent_by_ci95"] for row in rows
            ),
            "n_retained_after_holm_0_05": sum(
                row["retained_after_holm_0_05"] for row in rows
            ),
            "cells": rows,
        }
    return sweep


def inregion_iou(config: str, region: str) -> float:
    values = []
    for seed in range(10):
        if config == "unet":
            data = load_json(RESULTS / f"ftw_unet_{region}_seed{seed}.json")
            values.append(float(data[region]["unet_true_iou"]))
        else:
            mode = "frozen_decoder" if config == "frozen" else "full_finetune"
            data = load_json(
                RESULTS / f"ftw_finetune_fm_prithvi_{region}_{mode}_seed{seed}.json"
            )
            values.append(float(data[region]["ft_true_iou"]))
    return mean(values)


def cambodia_low_data() -> dict:
    output: dict[str, object] = {}
    for fraction in (0.1, 0.25, 0.5):
        auroc_values = []
        f1_values = []
        train_chips = set()
        for seed in range(3):
            data = load_json(
                RESULTS
                / f"ftw_finetune_fm_prithvi_camld_f{fraction}_s{seed}.json"
            )["cambodia"]
            auroc_values.append(float(data["ft_true_auroc"]))
            f1_values.append(float(data["ft_true_f1"]))
            train_chips.add(int(data["n_train_chips"]))
        if len(train_chips) != 1:
            raise ValueError(f"inconsistent Cambodia chip count at fraction {fraction}")
        output[f"{fraction:.2f}"] = {
            "n_seeds": 3,
            "n_train_chips": train_chips.pop(),
            "mean_auroc": round(mean(auroc_values), 4),
            "sample_std_auroc": round(sample_std(auroc_values), 4),
            "mean_f1": round(mean(f1_values), 4),
            "sample_std_f1": round(sample_std(f1_values), 4),
            "auroc_values": auroc_values,
            "f1_values": f1_values,
        }

    full_auroc = []
    full_f1 = []
    for seed in range(10):
        data = load_json(
            RESULTS
            / f"ftw_finetune_fm_prithvi_cambodia_full_finetune_seed{seed}.json"
        )["cambodia"]
        full_auroc.append(float(data["ft_true_auroc"]))
        full_f1.append(float(data["ft_true_f1"]))
    output["1.00"] = {
        "n_seeds": 10,
        "n_train_chips": 107,
        "mean_auroc": round(mean(full_auroc), 4),
        "sample_std_auroc": round(sample_std(full_auroc), 4),
        "mean_f1": round(mean(full_f1), 4),
        "sample_std_f1": round(sample_std(full_f1), 4),
        "auroc_values": full_auroc,
        "f1_values": full_f1,
    }
    return output


def build_regional_regime(inregion: dict) -> dict:
    regions: dict[str, object] = {}
    for region in REGIONS:
        split = load_json(RESULTS / f"ftw_split_{region}.json")
        regions[region] = {
            "positive_pixel_pct_rounded": POSITIVE_PIXEL_PCT[region],
            "n_train_chips": int(split["n_train_chips"]),
            "prithvi_frozen_minus_fullft_auroc": inregion["models"]["prithvi"][
                "regions"
            ][region]["paired_delta_mean"],
            "terramind_frozen_minus_fullft_auroc": inregion["models"][
                "terramind"
            ]["regions"][region]["paired_delta_mean"],
            "unet_mean_iou_at_0_5": round(inregion_iou("unet", region), 4),
            "prithvi_frozen_mean_iou_at_0_5": round(
                inregion_iou("frozen", region), 4
            ),
            "prithvi_fullft_mean_iou_at_0_5": round(
                inregion_iou("fullft", region), 4
            ),
            "descriptive_regime": REGIME_LABEL[region],
        }
    return {
        "method": (
            "Descriptive six-region summary. Training-chip counts, paired AUROC "
            "deltas, and ten-seed IoU means are regenerated from shipped JSONs. "
            "Positive-pixel percentages are rounded manuscript descriptors from "
            "the training masks; the raw masks are not redistributed. No causal "
            "threshold is estimated."
        ),
        "regions": regions,
        "cambodia_fullft_low_data_control": cambodia_low_data(),
    }


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> int:
    inregion = load_json(INREGION)
    write_json(EPS_OUTPUT, build_epsilon_sweep(inregion))
    write_json(REGIME_OUTPUT, build_regional_regime(inregion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
