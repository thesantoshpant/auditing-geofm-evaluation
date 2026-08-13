#!/usr/bin/env python
"""Aggregate the 80- and 150-epoch U-Net sensitivity runs.

The 80-epoch control contains three seeds (0--2). The primary 150-epoch
baseline contains ten seeds (0--9). The output is descriptive and is not a
seed-paired comparison because the two columns use different seed counts.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT = RESULTS / "ftw_unet_epoch_sensitivity.json"
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    output: dict[str, object] = {
        "method": (
            "Descriptive epoch-budget sensitivity. The 80-epoch control uses "
            "seeds 0--2; the primary 150-epoch baseline uses seeds 0--9. "
            "Differences are differences of means, not seed-paired estimates."
        ),
        "regions": {},
    }

    for region in REGIONS:
        values_80: list[float] = []
        for seed in range(3):
            data = load_json(RESULTS / f"ftw_unet_e80_seed{seed}.json")
            cell = data[region]
            if cell.get("seed") != seed or cell.get("epochs") != 80:
                raise ValueError(f"bad 80-epoch metadata for {region}, seed {seed}")
            values_80.append(float(cell["unet_true_auroc"]))

        values_150: list[float] = []
        for seed in range(10):
            data = load_json(RESULTS / f"ftw_unet_{region}_seed{seed}.json")
            cell = data[region]
            if cell.get("seed") != seed:
                raise ValueError(f"bad primary U-Net metadata for {region}, seed {seed}")
            values_150.append(float(cell["unet_true_auroc"]))

        mean_80 = statistics.fmean(values_80)
        mean_150 = statistics.fmean(values_150)
        output["regions"][region] = {
            "epochs_80": {
                "n_seeds": 3,
                "seeds": list(range(3)),
                "auroc_values": values_80,
                "mean_auroc": round(mean_80, 4),
                "sample_std_auroc": round(sample_std(values_80), 4),
            },
            "epochs_150": {
                "n_seeds": 10,
                "seeds": list(range(10)),
                "auroc_values": values_150,
                "mean_auroc": round(mean_150, 4),
                "sample_std_auroc": round(sample_std(values_150), 4),
            },
            "mean_delta_80_minus_150": round(mean_80 - mean_150, 4),
            "within_0_01_auroc": abs(mean_80 - mean_150) <= 0.01,
        }

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
