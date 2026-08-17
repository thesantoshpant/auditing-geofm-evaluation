#!/usr/bin/env python
"""Integrate the clean ten-seed FTW grid into canonical paper artifacts.

Inputs are the 300 raw headline-grid JSONs:
  - 2 GeoFMs x 6 regions x 2 modes x 10 seeds = 240 files
  - U-Net x 6 regions x 10 seeds = 60 files

Outputs:
  - data/results/ftw_inregion_equivalence.json
  - data/results/ftw_cross_region_transfer.json
  - data/results/ftw_headline_summary.txt

The script is intentionally strict: all 300 files must be present and each file
must contain all six target-region evaluations. Cross-region confidence
intervals are seed-paired: each seed contributes the mean delta over the selected
directed-transfer set.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover - scipy is present in the artifact env
    _stats = None

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]
MODELS = ["prithvi", "terramind"]
SEEDS = list(range(10))
EPS = 0.02
T_CRIT_9 = 2.2621571627409915


def mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def tcrit(df: int) -> float:
    if _stats is not None:
        return float(_stats.t.ppf(0.975, df))
    table = {9: T_CRIT_9}
    return table.get(df, 1.96)


def tsf(x: float, df: int) -> float:
    if _stats is not None:
        return float(_stats.t.sf(x, df))
    raise RuntimeError("scipy is required for TOST p-values")


def ci95(xs: list[float]) -> tuple[float, float]:
    h = tcrit(len(xs) - 1) * stdev(xs) / math.sqrt(len(xs))
    mu = mean(xs)
    return mu - h, mu + h


def round4(x: float) -> float:
    return round(float(x), 4)


def canonical_pvalue(value: float) -> float:
    """Serialize p-values at stable precision across supported SciPy releases."""
    return float(f"{value:.12g}")


def tost_pvalue(delta_mean: float, delta_sd: float, n: int, eps: float) -> float:
    if delta_sd == 0:
        return 0.0 if abs(delta_mean) < eps else 1.0
    se = delta_sd / math.sqrt(n)
    df = n - 1
    t_lower = (delta_mean - (-eps)) / se
    t_upper = (eps - delta_mean) / se
    # Use the survival function directly. Subtracting CDF values from one
    # loses precision in the small tails and produced version-dependent JSON.
    p_lower = tsf(t_lower, df)
    p_upper = tsf(t_upper, df)
    return max(p_lower, p_upper)


def holm_adjust(pvals: list[float]) -> list[float]:
    k = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda item: item[1])
    out = [0.0] * k
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adjusted = min(1.0, (k - rank) * p)
        running = max(running, adjusted)
        out[idx] = running
    return out


def transfer_key(source: str, target: str) -> str:
    return source if source == target else f"{source}->{target}"


def mode_file_tag(mode: str) -> str:
    return "frozen_decoder" if mode == "backbone" else "full_finetune"


def load_all() -> tuple[dict[tuple[str, str, str, str, int], dict[str, float]], dict[tuple[str, str, int], dict[str, float]]]:
    fm: dict[tuple[str, str, str, str, int], dict[str, float]] = {}
    unet: dict[tuple[str, str, int], dict[str, float]] = {}
    problems: list[str] = []

    for model in MODELS:
        for mode in ["backbone", "none"]:
            for source in REGIONS:
                for seed in SEEDS:
                    path = RESULTS / f"ftw_finetune_fm_{model}_{source}_{mode_file_tag(mode)}_seed{seed}.json"
                    if not path.exists():
                        problems.append(f"missing {path.relative_to(ROOT)}")
                        continue
                    data = json.loads(path.read_text())
                    expected = {transfer_key(source, target) for target in REGIONS}
                    if set(data) != expected:
                        problems.append(f"{path.name}: target keys differ")
                        continue
                    for target in REGIONS:
                        key = transfer_key(source, target)
                        cell = data[key]
                        if cell.get("train_country") != source or cell.get("eval_country") != target:
                            problems.append(f"{path.name}: bad train/eval metadata for {key}")
                        if cell.get("seed") != seed or cell.get("freeze") != mode:
                            problems.append(f"{path.name}: bad seed/freeze metadata for {key}")
                        fm[(model, mode, source, target, seed)] = {
                            metric: float(cell[f"ft_true_{metric}"])
                            for metric in ["auroc", "ap", "f1", "iou"]
                        }

    for source in REGIONS:
        for seed in SEEDS:
            path = RESULTS / f"ftw_unet_{source}_seed{seed}.json"
            if not path.exists():
                problems.append(f"missing {path.relative_to(ROOT)}")
                continue
            data = json.loads(path.read_text())
            expected = {transfer_key(source, target) for target in REGIONS}
            if set(data) != expected:
                problems.append(f"{path.name}: target keys differ")
                continue
            for target in REGIONS:
                key = transfer_key(source, target)
                cell = data[key]
                if cell.get("train_country") != source or cell.get("eval_country") != target:
                    problems.append(f"{path.name}: bad train/eval metadata for {key}")
                if cell.get("seed") != seed:
                    problems.append(f"{path.name}: bad seed metadata for {key}")
                unet[(source, target, seed)] = {
                    metric: float(cell[f"unet_true_{metric}"])
                    for metric in ["auroc", "ap", "f1", "iou"]
                }

    if problems:
        raise SystemExit("headline-grid validation failed:\n" + "\n".join(problems[:50]))
    return fm, unet


def inregion(fm: dict[tuple[str, str, str, str, int], dict[str, float]]) -> dict:
    ordered: list[tuple[str, str, float]] = []
    out: dict[str, object] = {
        "epsilon_auroc": EPS,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "method": (
            "Clean single-recipe 10-seed analysis. For each region x model "
            "cell, delta is frozen-backbone decoder AUROC minus full fine-tuning "
            "AUROC across the same ten seeds. TOST equivalence uses the conservative "
            "criterion that the 95% paired t-interval (df=9) lies fully within "
            "+/-0.02 AUROC. P-values are max of the two one-sided TOST p-values; "
            "Holm adjustment is across the 12 region x model cells. P-values are "
            "serialized to 12 significant digits to avoid library-level tail noise."
        ),
        "models": {},
    }
    total_equiv = 0

    for model in MODELS:
        model_out: dict[str, object] = {"regions": {}, "n_equivalent": 0}
        for region in REGIONS:
            frozen = [fm[(model, "backbone", region, region, seed)]["auroc"] for seed in SEEDS]
            fullft = [fm[(model, "none", region, region, seed)]["auroc"] for seed in SEEDS]
            deltas = [a - b for a, b in zip(frozen, fullft, strict=True)]
            lo, hi = ci95(deltas)
            delta_mean = mean(deltas)
            delta_sd = stdev(deltas)
            tost = lo >= -EPS and hi <= EPS
            p = canonical_pvalue(tost_pvalue(delta_mean, delta_sd, len(SEEDS), EPS))
            cell = {
                "frozen_seeds": frozen,
                "fullft_seeds": fullft,
                "frozen_mean": round4(mean(frozen)),
                "frozen_std": round4(stdev(frozen)),
                "fullft_mean": round4(mean(fullft)),
                "fullft_std": round4(stdev(fullft)),
                "paired_delta_mean": round4(delta_mean),
                "paired_delta_std": round4(delta_sd),
                "ci95": [round4(lo), round4(hi)],
                "tost_equivalent": tost,
                "tost_p_value": p,
                "tost_p_value_display": round(p, 6),
            }
            model_out["regions"][region] = cell
            ordered.append((model, region, p))
            if tost:
                model_out["n_equivalent"] += 1
                total_equiv += 1
        out["models"][model] = model_out

    adjusted = [canonical_pvalue(value) for value in holm_adjust([p for _, _, p in ordered])]
    for (model, region, _), adj in zip(ordered, adjusted, strict=True):
        cell = out["models"][model]["regions"][region]
        cell["tost_p_value_holm"] = adj
        cell["tost_p_value_holm_display"] = round(adj, 6)
        cell["tost_equiv_holm_0_05"] = adj < 0.05
        cell["tost_equiv_holm_0_025"] = adj < 0.025

    out["n_equivalent_combined"] = total_equiv
    out["n_total_cells"] = len(MODELS) * len(REGIONS)
    out["holm"] = {
        "n_cells": len(ordered),
        "n_retained_alpha_0_05": sum(
            1
            for model in MODELS
            for region in REGIONS
            if out["models"][model]["regions"][region]["tost_equiv_holm_0_05"]
        ),
        "n_retained_alpha_0_025": sum(
            1
            for model in MODELS
            for region in REGIONS
            if out["models"][model]["regions"][region]["tost_equiv_holm_0_025"]
        ),
    }
    return out


def get_value(
    label: str,
    source: str,
    target: str,
    seed: int,
    fm: dict[tuple[str, str, str, str, int], dict[str, float]],
    unet: dict[tuple[str, str, int], dict[str, float]],
    metric: str = "auroc",
) -> float:
    if label == "unet":
        return unet[(source, target, seed)][metric]
    model, mode = label.split(":")
    return fm[(model, mode, source, target, seed)][metric]


def transfer(fm: dict[tuple[str, str, str, str, int], dict[str, float]], unet: dict[tuple[str, str, int], dict[str, float]]) -> dict:
    all30 = [(a, b) for a in REGIONS for b in REGIONS if a != b]
    nonkenya20 = [(a, b) for a, b in all30 if a != "kenya" and b != "kenya"]
    kenya10 = [(a, b) for a, b in all30 if a == "kenya" or b == "kenya"]
    configs = {
        "unet": "unet",
        "prithvi_frozen": "prithvi:backbone",
        "prithvi_fullft": "prithvi:none",
        "terramind_frozen": "terramind:backbone",
        "terramind_fullft": "terramind:none",
    }
    comparisons = [
        ("prithvi_frozen", "prithvi_fullft"),
        ("terramind_frozen", "terramind_fullft"),
        ("prithvi_frozen", "unet"),
        ("terramind_frozen", "unet"),
        ("prithvi_fullft", "unet"),
        ("terramind_fullft", "unet"),
    ]

    out: dict[str, object] = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "method": (
            "Clean 10-seed cross-region analysis over directed train-country -> "
            "test-country transfers. Confidence intervals are seed-paired: each "
            "seed contributes one mean delta over the directed-transfer subset "
            "(all30, nonkenya20, or kenya10), then a t-interval is computed over "
            "the ten seed-level means."
        ),
        "configs": list(configs),
        "directed_pairs": {
            "all30": [f"{a}->{b}" for a, b in all30],
            "nonkenya20": [f"{a}->{b}" for a, b in nonkenya20],
            "kenya10": [f"{a}->{b}" for a, b in kenya10],
        },
        "pairs": {},
        "subsets": {},
    }

    for source, target in all30:
        pair_key = f"{source}->{target}"
        row = {}
        for cfg_name, cfg_label in configs.items():
            vals = [get_value(cfg_label, source, target, seed, fm, unet) for seed in SEEDS]
            row[cfg_name] = {
                "mean_auroc": round4(mean(vals)),
                "std_auroc": round4(stdev(vals)),
                "seeds": vals,
            }
        out["pairs"][pair_key] = row

    for subset_name, subset in [("all30", all30), ("nonkenya20", nonkenya20), ("kenya10", kenya10)]:
        subset_out = {"n_pairs": len(subset), "means": {}, "seed_paired_deltas": {}}
        for cfg_name, cfg_label in configs.items():
            vals = [
                get_value(cfg_label, source, target, seed, fm, unet)
                for source, target in subset
                for seed in SEEDS
            ]
            subset_out["means"][cfg_name] = round4(mean(vals))

        for a, b in comparisons:
            seed_means = []
            for seed in SEEDS:
                deltas = [
                    get_value(configs[a], source, target, seed, fm, unet)
                    - get_value(configs[b], source, target, seed, fm, unet)
                    for source, target in subset
                ]
                seed_means.append(mean(deltas))
            lo, hi = ci95(seed_means)
            subset_out["seed_paired_deltas"][f"{a}_minus_{b}"] = {
                "seed_mean_deltas": seed_means,
                "mean_delta": round4(mean(seed_means)),
                "std_delta": round4(stdev(seed_means)),
                "ci95": [round4(lo), round4(hi)],
            }
        out["subsets"][subset_name] = subset_out
    return out


def write_summary(inreg: dict, xfer: dict) -> str:
    lines: list[str] = []
    lines.append("FTW clean single-recipe 10-seed summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append(
        f"In-region TOST: {inreg['n_equivalent_combined']} of {inreg['n_total_cells']} "
        f"region x model cells equivalent at epsilon={EPS} AUROC (n=10, df=9)."
    )
    for model in MODELS:
        lines.append(f"  {model}: {inreg['models'][model]['n_equivalent']} of 6")
    lines.append("")
    lines.append("In-region cells (delta = frozen - full-FT):")
    for model in MODELS:
        for region in REGIONS:
            cell = inreg["models"][model]["regions"][region]
            verdict = "equiv" if cell["tost_equivalent"] else "not-equiv"
            lines.append(
                f"  {model:<9} {region:<11} frozen={cell['frozen_mean']:.4f} "
                f"fullft={cell['fullft_mean']:.4f} delta={cell['paired_delta_mean']:+.4f} "
                f"CI=[{cell['ci95'][0]:+.4f},{cell['ci95'][1]:+.4f}] {verdict}"
            )
    lines.append("")
    lines.append("Cross-region subset means:")
    for subset_name in ["all30", "nonkenya20", "kenya10"]:
        subset = xfer["subsets"][subset_name]
        means = subset["means"]
        lines.append(
            f"  {subset_name:<11} U-Net={means['unet']:.4f} "
            f"Prithvi frozen={means['prithvi_frozen']:.4f} "
            f"Prithvi fullFT={means['prithvi_fullft']:.4f} "
            f"TerraMind frozen={means['terramind_frozen']:.4f} "
            f"TerraMind fullFT={means['terramind_fullft']:.4f}"
        )
    lines.append("")
    lines.append("Cross-region seed-paired deltas:")
    for subset_name in ["all30", "nonkenya20"]:
        lines.append(f"  {subset_name}:")
        deltas = xfer["subsets"][subset_name]["seed_paired_deltas"]
        for key in [
            "prithvi_frozen_minus_prithvi_fullft",
            "terramind_frozen_minus_terramind_fullft",
            "prithvi_frozen_minus_unet",
            "terramind_frozen_minus_unet",
        ]:
            d = deltas[key]
            lines.append(
                f"    {key}: delta={d['mean_delta']:+.4f} "
                f"CI=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    fm, unet = load_all()
    inreg = inregion(fm)
    xfer = transfer(fm, unet)
    summary = write_summary(inreg, xfer)

    with (RESULTS / "ftw_inregion_equivalence.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(inreg, handle, indent=2)
    with (RESULTS / "ftw_cross_region_transfer.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(xfer, handle, indent=2)
    with (RESULTS / "ftw_headline_summary.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(summary)
    print(summary)
    print("Wrote data/results/ftw_inregion_equivalence.json")
    print("Wrote data/results/ftw_cross_region_transfer.json")
    print("Wrote data/results/ftw_headline_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
