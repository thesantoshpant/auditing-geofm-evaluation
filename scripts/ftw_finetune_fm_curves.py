#!/usr/bin/env python
"""Summarize and plot the 150-epoch Prithvi convergence diagnostic.

The saved trajectories are post-hoc test-set evaluations from seed 0. They were
not used to choose checkpoints or hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
SUMMARY = RESULTS / "ftw_convergence_diagnostic.json"
FIGURE = ROOT / "paper" / "figures" / "convergence.pdf"
PDF_METADATA = {
    "Creator": "Auditing GeoFM Evaluation artifact",
    "Producer": "Auditing GeoFM Evaluation artifact",
    "CreationDate": None,
    "ModDate": None,
}
REGIONS = ("vietnam", "kenya")
CONFIGS = {
    "frozen_decoder": "ftw_finetune_fm_prithvi_curve150_backbone_seed0.json",
    "full_finetune": "ftw_finetune_fm_prithvi_curve150_none_seed0.json",
}


def load_result(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    for region in REGIONS:
        cell = result.get(region, {})
        if cell.get("seed") != 0 or cell.get("epochs") != 150:
            raise ValueError(f"bad curve metadata in {path.name}: {region}")
        epochs = [point["epoch"] for point in cell.get("epoch_curve", [])]
        if epochs != list(range(15, 151, 15)):
            raise ValueError(f"unexpected epoch sequence in {path.name}: {region}")
    return result


def summarize_curve(points: list[dict]) -> dict:
    values = [float(point["auroc"]) for point in points]
    peak_index = max(range(len(points)), key=lambda index: values[index])
    return {
        "points": points,
        "peak_epoch": int(points[peak_index]["epoch"]),
        "peak_auroc": values[peak_index],
        "final_auroc": values[-1],
        "range_auroc": round(max(values) - min(values), 4),
        "epoch_15_auroc": values[0],
        "epoch_75_auroc": values[4],
        "epoch_90_auroc": values[5],
        "epoch_150_auroc": values[-1],
    }


def main() -> int:
    raw = {config: load_result(name) for config, name in CONFIGS.items()}
    summary: dict[str, object] = {
        "method": (
            "Post-hoc seed-0 test-set trajectories recorded every 15 epochs. "
            "The trajectories were not used for model or checkpoint selection; "
            "the primary experiments select the last epoch without early stopping."
        ),
        "primary_epoch_budget": 80,
        "regions": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"frozen_decoder": "#2b6cb0", "full_finetune": "#c53030"}
    markers = {"frozen_decoder": "o", "full_finetune": "s"}
    labels = {"frozen_decoder": "frozen decoder", "full_finetune": "full fine-tuning"}

    for axis, region in zip(axes, REGIONS, strict=True):
        region_summary: dict[str, object] = {}
        for config in CONFIGS:
            points = raw[config][region]["epoch_curve"]
            region_summary[config] = summarize_curve(points)
            axis.plot(
                [point["epoch"] for point in points],
                [point["auroc"] for point in points],
                marker=markers[config],
                label=labels[config],
                color=colors[config],
                linewidth=2,
            )
        summary["regions"][region] = region_summary
        axis.axvline(
            80,
            color="grey",
            linestyle="--",
            alpha=0.7,
            label="80 epochs (primary budget)",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Test-set AUROC")
        axis.set_title(region.title())
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    fig.suptitle("Prithvi convergence across 150 epochs (single seed)", fontsize=11)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)

    with SUMMARY.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {FIGURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
