"""Seed-paired TOST analysis for TerraMind frozen-vs-FT across 6 regions.

Uses canonical single-pipeline sources per (region, mode):
  frozen:  probeIR_bb (existing 3-seed) for India, Cambodia, France;
           v7 runs for Vietnam, Kenya, Netherlands.
  FT:      seedD (existing 3-seed) for all 5 regions where it exists;
           v7 runs for Cambodia (the missing cell).

This avoids the recipe-mixing the prior v7 paper review caught.
"""
import json
import os
import statistics
from math import sqrt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")

REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]
EPS = 0.02
T_DF2 = 4.303  # 95% CI t at df=2 (3 seeds, paired)


def get_seed(region, mode, seed):
    if mode == "frozen":
        p = os.path.join(R, f"ftw_finetune_fm_terramind_probeIR_bb_s{seed}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if region in d:
                return d[region]["ft_true_auroc"]
        suffix = "" if seed == 0 else f"_s{seed}"
        p = os.path.join(R, f"ftw_finetune_fm_terramindterramind_v7_{region}_frozen{suffix}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if region in d:
                return d[region]["ft_true_auroc"]
    else:
        p = os.path.join(R, f"ftw_finetune_fm_terramind_seedD_s{seed}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if region in d:
                return d[region]["ft_true_auroc"]
        suffix = "" if seed == 0 else f"_s{seed}"
        p = os.path.join(R, f"ftw_finetune_fm_terramindterramind_v7_{region}_ft{suffix}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if region in d:
                return d[region]["ft_true_auroc"]
    return None


def main():
    print(f"{'Region':<13} {'frozen seeds':<30} {'FT seeds':<30} {'paired delta':<20} {'95% CI':<22} TOST")
    print("-" * 130)
    eq = 0
    rows = {}
    for r in REGIONS:
        fb = [get_seed(r, "frozen", s) for s in [0, 1, 2]]
        ft = [get_seed(r, "ft", s) for s in [0, 1, 2]]
        if None in fb or None in ft:
            print(f"{r:<13} MISSING: frozen={fb} ft={ft}")
            continue
        diffs = [fb[i] - ft[i] for i in range(3)]
        mean_d = statistics.mean(diffs)
        std_d = statistics.stdev(diffs)
        se = std_d / sqrt(3)
        half = T_DF2 * se
        ci_lo = mean_d - half
        ci_hi = mean_d + half
        tost = ci_lo > -EPS and ci_hi < EPS
        verdict = "EQUIV" if tost else "not-equiv"
        fb_s = "[" + ",".join(f"{v:.4f}" for v in fb) + "]"
        ft_s = "[" + ",".join(f"{v:.4f}" for v in ft) + "]"
        ci_s = f"[{ci_lo:+.4f},{ci_hi:+.4f}]"
        d_s = f"{mean_d:+.4f}+-{std_d:.4f}"
        print(f"{r:<13} {fb_s:<30} {ft_s:<30} {d_s:<20} {ci_s:<22} {verdict}")
        rows[r] = {"frozen_seeds": fb, "ft_seeds": ft, "mean_d": mean_d, "std_d": std_d,
                   "ci_lo": ci_lo, "ci_hi": ci_hi, "tost_equivalent": tost}
        if tost:
            eq += 1

    print(f"\nTerraMind paired TOST at eps={EPS}: {eq} of {len(rows)} regions equivalent\n")

    # Save to JSON for the paper
    out = os.path.join(R, "ftw_terramind_paired_tost.json")
    json.dump({"epsilon_auroc": EPS, "t_crit_df2": T_DF2, "method": "seed-paired t-interval, df=2",
               "regions": rows, "n_equivalent": eq, "n_total": len(rows)}, open(out, "w"), indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
