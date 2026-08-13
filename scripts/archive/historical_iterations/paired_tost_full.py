"""Seed-paired TOST analysis for Prithvi AND TerraMind frozen-vs-FT across 6 regions.

Source-resolution rule (strict family-level):
For each (model, region, mode) cell, pick the FIRST source family that has ALL seeds
(seeds 0..N-1) present. No mixing seeds across families within a cell.

Resolution order:
  (1) reproduce_all.sh tag family   (so a full rerun drives the headline)
  (2) v7 tag family                  (the round-3 single-region runs)
  (3) canonical multi-region family  (the original shipped sprint runs)

Default seed budget is 3 (s0..s2). Use --max-seed N to extend to s0..s(N-1) once
the v8 grid (seeds 3..9) lands; combined with seeds 0..2 from canonical sprints, the cohort is mixed-recipe and therefore not run end-to-end here -- use scripts/paired_tost_v8.py for the pure n=7 v8 analysis instead.
"""
import argparse
import json
import os
import statistics
from math import sqrt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]
EPS = 0.02
T_TABLE = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def read(fname, region):
    p = os.path.join(R, fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if region in d:
        return d[region].get("ft_true_auroc")
    return None


def family_complete(seeds_dict):
    """All seeds present (no None) and at least 3."""
    return all(v is not None for v in seeds_dict.values()) and len(seeds_dict) >= 3


def collect_family(model, region, mode, seeds, family):
    """family in {'v8', 'reproduce', 'v7', 'canonical'}. Returns dict {seed: auroc or None}."""
    out = {}
    for s in seeds:
        if family == "v8":
            # v8 grid (seeds 3..9): tags are <model>_v8_<region>_<mode>_s<seed>
            mode_word = "backbone" if mode == "frozen" else "none"
            if model == "prithvi":
                fn = f"ftw_finetune_fm_prithviprithvi_v8_{region}_{mode_word}_s{s}.json"
            else:
                fn = f"ftw_finetune_fm_terramindterramind_v8_{region}_{mode_word}_s{s}.json"
            out[s] = read(fn, region)
        elif family == "reproduce":
            if model == "prithvi":
                fn = f"ftw_finetune_fm_prithviprithvi_{region}_{('frozen' if mode == 'frozen' else 'ft')}_s{s}.json"
            else:
                fn = f"ftw_finetune_fm_terramindterramind_{region}_{('frozen' if mode == 'frozen' else 'ft')}_s{s}.json"
            out[s] = read(fn, region)
        elif family == "v7":
            if model == "prithvi":
                out[s] = None  # no v7-family runs for Prithvi
                continue
            suffix = "" if s == 0 else f"_s{s}"
            fn = f"ftw_finetune_fm_terramindterramind_v7_{region}_{('frozen' if mode == 'frozen' else 'ft')}{suffix}.json"
            out[s] = read(fn, region)
        else:  # canonical
            if model == "prithvi" and mode == "frozen":
                fn = f"ftw_finetune_fm_prithvi_probe_backbone_s{s}.json" if region == "india" \
                     else f"ftw_finetune_fm_prithvi_probeIR_bb_s{s}.json"
            elif model == "prithvi" and mode == "ft":
                if region in ("india", "vietnam", "kenya"):
                    fn = f"ftw_finetune_fm_prithvi_ld_f1.0_s{s}.json"
                elif region == "cambodia":
                    fn = f"ftw_finetune_fm_prithvi_camB_s{s}.json"
                else:
                    fn = f"ftw_finetune_fm_prithvi_seedB_s{s}.json"
            elif model == "terramind" and mode == "frozen":
                fn = f"ftw_finetune_fm_terramind_probeIR_bb_s{s}.json"
            else:  # terramind ft
                fn = f"ftw_finetune_fm_terramind_seedD_s{s}.json"
            out[s] = read(fn, region)
    return out


def resolve(model, region, mode, seeds):
    """Return (values_list, family_name) where family is the first complete one.
    Order: v8 grid (covers seeds 3..9 only), reproduce_all.sh tag, v7 tag, canonical."""
    for family in ("v8", "reproduce", "v7", "canonical"):
        d = collect_family(model, region, mode, seeds, family)
        if family_complete(d):
            return [d[s] for s in seeds], family
    return None, None


def paired_tost(model, seeds):
    n = len(seeds)
    t_crit = T_TABLE.get(n - 1, 1.96)
    print(f"\n--- {model.title()} (seed-paired t-interval, df={n-1}, n={n} seeds) ---")
    print(f"{'Region':<13} {'family':<10} {'frozen':<30} {'FT':<30} {'paired delta':<22} {'95% CI':<24} TOST")
    print("-" * 145)
    out = {}
    eq = 0
    for r in REGIONS:
        fb, fam_fb = resolve(model, r, "frozen", seeds)
        ft, fam_ft = resolve(model, r, "ft", seeds)
        if fb is None or ft is None:
            print(f"{r:<13} MISSING: frozen_family={fam_fb} ft_family={fam_ft}")
            continue
        diffs = [fb[i] - ft[i] for i in range(n)]
        m = statistics.mean(diffs)
        s = statistics.stdev(diffs) if n > 1 else 0.0
        se = s / sqrt(n)
        half = t_crit * se
        lo, hi = m - half, m + half
        tost = lo > -EPS and hi < EPS
        verdict = "EQUIV" if tost else "not-eq"
        fam_str = f"{fam_fb[:5]}/{fam_ft[:5]}"
        fb_s = "[" + ",".join(f"{v:.4f}" for v in fb[:3]) + ("...]" if len(fb) > 3 else "]")
        ft_s = "[" + ",".join(f"{v:.4f}" for v in ft[:3]) + ("...]" if len(ft) > 3 else "]")
        ci_s = f"[{lo:+.4f},{hi:+.4f}]"
        d_s = f"{m:+.4f}+-{s:.4f}"
        print(f"{r:<13} {fam_str:<10} {fb_s:<30} {ft_s:<30} {d_s:<22} {ci_s:<24} {verdict}")
        out[r] = {"frozen_seeds": fb, "ft_seeds": ft, "frozen_source_family": fam_fb,
                  "ft_source_family": fam_ft,
                  "paired_delta_mean": round(m, 4), "paired_delta_std": round(s, 4),
                  "ci": [round(lo, 4), round(hi, 4)], "tost_equivalent": tost}
        if tost:
            eq += 1
    print(f"\n{model.title()}: {eq} of {len(out)} regions TOST-equivalent at epsilon={EPS}")
    return out, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seed", type=int, default=3,
                    help="Use seeds 0..max_seed-1. Default 3 (s0,s1,s2) -- the "
                         "originally specified analysis. Note: the v8 grid populates "
                         "seeds 3..9 only (not 0..2), so this script will fail-fast at "
                         "--max-seed > 3; for the seven-seed v8 cohort, use the "
                         "dedicated scripts/paired_tost_v8.py instead.")
    a = ap.parse_args()
    seeds = list(range(a.max_seed))
    pri_out, pri_eq = paired_tost("prithvi", seeds)
    tm_out, tm_eq = paired_tost("terramind", seeds)
    if len(pri_out) != 6 or len(tm_out) != 6:
        import sys
        sys.stderr.write(
            f"\nERROR: incomplete grid (prithvi {len(pri_out)}/6 cells, terramind "
            f"{len(tm_out)}/6 cells). paired_tost_full.py requires all 12 (model, "
            f"region, mode) arms to be complete in some source family. Re-run with "
            f"the correct --max-seed (you used {a.max_seed}); note the v8 grid "
            f"covers seeds 3..9 only. For the n=7 v8 cohort use paired_tost_v8.py.\n"
        )
        sys.exit(2)
    total = pri_eq + tm_eq
    print(f"\n=== COMBINED: {total} of 12 region x model cells equivalent at epsilon={EPS} (n={len(seeds)} seed-paired) ===")
    out = os.path.join(R, "ftw_paired_tost.json")
    json.dump({"epsilon_auroc": EPS, "n_seeds": len(seeds), "t_crit": T_TABLE.get(len(seeds) - 1, 1.96),
               "method": f"seed-paired t-interval (df={len(seeds)-1}); strict source-family resolution (reproduce > v7 > canonical, all seeds from same family within a cell)",
               "prithvi": {"regions": pri_out, "n_equivalent": pri_eq},
               "terramind": {"regions": tm_out, "n_equivalent": tm_eq},
               "n_equivalent_combined": total, "n_total_cells": 12},
              open(out, "w"), indent=2)
    print(f"\nWrote data/results/{os.path.basename(out)}")


if __name__ == "__main__":
    main()
