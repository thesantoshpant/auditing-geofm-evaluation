"""Aggregate per-region n-seed mean+/-std for frozen-decoder and full-FT
(Prithvi + TerraMind) into ftw_probe_inregion.json.

Uses the same strict source-family resolution as paired_tost_full.py: for each (model,
region, mode) cell, picks the FIRST family that has all required seeds present, no
seed mixing within a cell. Order: reproduce_all.sh tag > v7 tag > canonical.

Default seed budget is 3. Use --max-seed N to scale to s0..s(N-1) once v8 lands.
"""
import argparse
import json
import os
import statistics
import sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]


def read(fname, region):
    p = os.path.join(R, fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if region in d:
        return d[region].get("ft_true_auroc")
    return None


def collect(model, region, mode, family, seeds):
    out = {}
    for s in seeds:
        if family == "v8":
            mode_word = "backbone" if mode == "frozen" else "none"
            fn = (f"ftw_finetune_fm_prithviprithvi_v8_{region}_{mode_word}_s{s}.json"
                  if model == "prithvi" else
                  f"ftw_finetune_fm_terramindterramind_v8_{region}_{mode_word}_s{s}.json")
            out[s] = read(fn, region)
        elif family == "reproduce":
            stub = "frozen" if mode == "frozen" else "ft"
            fn = (f"ftw_finetune_fm_prithviprithvi_{region}_{stub}_s{s}.json"
                  if model == "prithvi" else
                  f"ftw_finetune_fm_terramindterramind_{region}_{stub}_s{s}.json")
            out[s] = read(fn, region)
        elif family == "v7":
            if model == "prithvi":
                out[s] = None
                continue
            stub = "frozen" if mode == "frozen" else "ft"
            suffix = "" if s == 0 else f"_s{s}"
            fn = f"ftw_finetune_fm_terramindterramind_v7_{region}_{stub}{suffix}.json"
            out[s] = read(fn, region)
        else:  # canonical
            if model == "prithvi" and mode == "frozen":
                fn = (f"ftw_finetune_fm_prithvi_probe_backbone_s{s}.json" if region == "india"
                      else f"ftw_finetune_fm_prithvi_probeIR_bb_s{s}.json")
            elif model == "prithvi" and mode == "ft":
                if region in ("india", "vietnam", "kenya"):
                    fn = f"ftw_finetune_fm_prithvi_ld_f1.0_s{s}.json"
                elif region == "cambodia":
                    fn = f"ftw_finetune_fm_prithvi_camB_s{s}.json"
                else:
                    fn = f"ftw_finetune_fm_prithvi_seedB_s{s}.json"
            elif model == "terramind" and mode == "frozen":
                fn = f"ftw_finetune_fm_terramind_probeIR_bb_s{s}.json"
            else:
                fn = f"ftw_finetune_fm_terramind_seedD_s{s}.json"
            out[s] = read(fn, region)
    return out


def resolve(model, region, mode, seeds):
    """Order: v8 grid > reproduce_all.sh > v7 single-region > canonical multi-region."""
    for family in ("v8", "reproduce", "v7", "canonical"):
        d = collect(model, region, mode, family, seeds)
        if all(v is not None for v in d.values()):
            return [d[s] for s in seeds], family
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seed", type=int, default=3,
                    help="Use seeds 0..max_seed-1. Default 3 (the originally "
                         "specified analysis). The v8 grid populates seeds 3..9 only, "
                         "so this script will fail-fast at --max-seed > 3; for the "
                         "seven-seed v8 cohort see scripts/paired_tost_v8.py.")
    a = ap.parse_args()
    seeds = list(range(a.max_seed))
    out = {"prithvi": {}, "terramind": {}}
    incomplete = []
    for model in ("prithvi", "terramind"):
        for r in REGIONS:
            fb, fb_fam = resolve(model, r, "frozen", seeds)
            ft, ft_fam = resolve(model, r, "ft", seeds)
            if fb is None or ft is None:
                incomplete.append(f"{model}/{r}: frozen_family={fb_fam} ft_family={ft_fam}")
                continue
            paired = [fb[i] - ft[i] for i in range(len(seeds))]
            out[model][r] = {
                "frozen_backbone": [round(statistics.mean(fb), 4), round(statistics.stdev(fb), 4)],
                "full_ft": [round(statistics.mean(ft), 4), round(statistics.stdev(ft), 4)],
                "delta_frozen_minus_full": round(statistics.mean(paired), 4),
                "paired_delta_std": round(statistics.stdev(paired), 4),
                "seeds_frozen": fb,
                "seeds_ft": ft,
                "frozen_source_family": fb_fam,
                "ft_source_family": ft_fam,
            }

    if incomplete:
        sys.stderr.write(
            f"\nERROR: incomplete grid for {len(incomplete)} of 12 cells at "
            f"--max-seed={a.max_seed}:\n  " + "\n  ".join(incomplete) +
            f"\n\nintegrate_paired.py requires all 12 cells to resolve. "
            f"Re-run with --max-seed=3 (only complete configuration). The v8 grid "
            f"covers seeds 3..9 only; for the n=7 v8 cohort see paired_tost_v8.py.\n"
        )
        sys.exit(2)
    target = os.path.join(R, "ftw_probe_inregion.json")
    with open(target, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote data/results/{os.path.basename(target)} (n_seeds={len(seeds)})")
    for model in out:
        for r in REGIONS:
            v = out[model].get(r)
            if not v:
                continue
            print(f"  {model:<10} {r:<13} fb={v['frozen_backbone']} ft={v['full_ft']} "
                  f"d={v['delta_frozen_minus_full']:+.4f}+-{v['paired_delta_std']:.4f} "
                  f"({v['frozen_source_family'][:5]}/{v['ft_source_family'][:5]})")


if __name__ == "__main__":
    main()
