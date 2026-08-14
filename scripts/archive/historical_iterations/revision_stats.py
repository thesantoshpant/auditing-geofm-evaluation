"""Historical, non-executable snapshot of the pre-camera-ready revision helper.

Paths and filenames below intentionally preserve the old release layout. Use
the active integration scripts at ``scripts/`` for the camera-ready artifact.

Outputs LaTeX-ready fragments to stdout grouped by table.

Data sources:
- ftw_bootstrap_ci_{region}.json  -> chip-level bootstrap CIs (B=1000) for
  frozen-probe models {Clay, Prithvi, TerraMind, AnySat, RF} on AUROC and F1.
  Available for India, Kenya, Vietnam, France, Netherlands.
- ftw_master_results.json  -> per-region [mean, std] over 3 seeds for U-Net,
  Prithvi-ft. Includes Cambodia.
- ftw_probe_inregion.json  -> per-region {frozen_backbone, full_ft} = [mean, std]
  over 3 seeds for Prithvi and partial TerraMind.
- ftw_transfer_strengthened.json -> per-transfer [mean, std] over 3 seeds for
  U-Net / Prithvi-ft / frozen-{Prithvi,TerraMind,Clay} / spectral-RF.

Statistical methods:
- For frozen-probes: report point + chip-level 95% CI from bootstrap files.
- For U-Net / Prithvi-ft / frozen-decoder: report mean +- seed-std bounds
  (95% via t-distribution with df=2 since only 3 seeds; this is honest about
  the thin uncertainty).
- TOST equivalence: for "frozen ~ fine-tune" claims, declare equivalence margin
  eps and test whether the seed-CI on the difference fully lies within +-eps.
"""
import json
from pathlib import Path
from math import sqrt

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

# t critical value with df=2 (3 seeds -> n-1=2 df) for two-sided 95% CI = 4.303
T_CRIT_2DF_95 = 4.303

REGIONS = ["India", "Cambodia", "Vietnam", "Kenya", "France", "Netherlands"]
REGION_KEYS = {r: r.lower() for r in REGIONS}


def t_ci_from_seeds(mean, std, n=3, t_crit=T_CRIT_2DF_95):
    """Return (low, high) 95% CI from mean +- std over n seeds via t-distribution."""
    se = std / sqrt(n)
    halfwidth = t_crit * se
    return (mean - halfwidth, mean + halfwidth)


def fmt_ci(point, lo, hi):
    return f"{point:.3f}\\,{{\\scriptsize[{lo:.3f},\\,{hi:.3f}]}}"


def fmt_ms(mean, std):
    return f"{mean:.3f}\\,{{\\scriptsize$\\pm$.{int(round(std*1000)):03d}}}"


def load_bootstrap(region):
    path = RESULTS / f"ftw_bootstrap_ci_{region.lower()}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("# Phase B statistics — generated from existing JSONs")
    print()

    # -----------------------------------------------------------------------
    # Master results: per-region U-Net seeds, Prithvi-ft seeds, RF, best frozen
    # -----------------------------------------------------------------------
    with open(RESULTS / "ftw_master_results.json") as f:
        master = json.load(f)

    print("## Table 1 augmentation: per-region point + 95% CI")
    print()
    print(f"{'Region':<12} {'RF-true':<22} {'U-Net (seed-CI)':<32} {'Frozen-probe Prithvi (chip-CI)':<42} {'Full-FT (seed-CI)':<32}")
    print('-' * 145)
    for region in REGIONS:
        key = REGION_KEYS[region]
        m = master.get(key, {})
        rf = m.get('rf_true', None)
        unet = m.get('unet')
        ft = m.get('prithvi_ft')

        boot = load_bootstrap(region)
        if boot:
            p = boot['models']['prithvi_eo_2_0_300m']['true']['auroc']
            pp_str = fmt_ci(p['point'], p['ci'][0], p['ci'][1])
        else:
            pp_str = "n/a"

        u_str = "n/a"
        if unet:
            u_lo, u_hi = t_ci_from_seeds(unet[0], unet[1])
            u_str = f"{unet[0]:.3f}[{u_lo:.3f},{u_hi:.3f}]"
        ft_str = "n/a"
        if ft:
            f_lo, f_hi = t_ci_from_seeds(ft[0], ft[1])
            ft_str = f"{ft[0]:.3f}[{f_lo:.3f},{f_hi:.3f}]"
        print(f"{region:<12} {str(rf):<22} {u_str:<32} {pp_str:<42} {ft_str:<32}")

    print()
    print("## Table 2 augmentation: frozen-decoder vs full-FT (Prithvi) with TOST")
    print()
    # Consume the canonical seed-paired TOST output (ftw_paired_tost.json) so this
    # summary cannot drift from the headline analysis. paired_tost_full.py computes
    # the CI from matched per-seed differences, not pooled independent arms.
    with open(RESULTS / "ftw_paired_tost.json") as f:
        ptost = json.load(f)
    prithvi_regions = ptost['prithvi']['regions']

    EPS_AUROC = ptost['epsilon_auroc']
    print(f"Equivalence margin epsilon = {EPS_AUROC:.2f} AUROC  (paired CI; n=3, df=2)")
    print()
    print(f"{'Region':<12} {'frozen-dec':<20} {'full-FT':<20} {'delta':<10} {'paired CI':<22} {'TOST verdict':<18}")
    print('-' * 105)
    eq_count = 0
    total = 0
    for region in REGIONS:
        key = REGION_KEYS[region]
        if key not in prithvi_regions:
            continue
        r = prithvi_regions[key]
        fb_seeds = r['frozen_seeds']
        ft_seeds = r['ft_seeds']
        fb_mean = sum(fb_seeds) / len(fb_seeds)
        ft_mean = sum(ft_seeds) / len(ft_seeds)
        delta = r['paired_delta_mean']
        ci_lo, ci_hi = r['ci']
        equivalent = r['tost_equivalent']
        verdict = "equivalent" if equivalent else ("different" if (ci_lo > 0 or ci_hi < 0) else "inconclusive")
        if equivalent:
            eq_count += 1
        total += 1
        d_ci = f"[{ci_lo:+.3f},{ci_hi:+.3f}]"
        print(f"{region:<12} {fb_mean:.3f}            {ft_mean:.3f}            {delta:+.3f}    {d_ci:<22} {verdict:<18}")
    print(f"\nTOST result: {eq_count}/{total} regions equivalent within +-{EPS_AUROC} AUROC (seed-paired, Prithvi only)")

    # -----------------------------------------------------------------------
    # Table 5 — cross-region transfer with paired delta and seed CIs
    # -----------------------------------------------------------------------
    print()
    print("## Table 5 augmentation: cross-region transfer means with seed CIs")
    print()
    with open(RESULTS / "ftw_transfer_strengthened.json") as f:
        xfer = json.load(f)
    pairs = xfer['pairs']
    models = ['unet_cross', 'prithvi_ft_cross', 'frozen_prithvi_cross',
              'frozen_terramind_cross', 'frozen_clay_cross', 'spectral_rf_cross']
    all_keys = list(pairs.keys())
    nonkenya = [k for k in all_keys if 'kenya' not in k]

    print(f"{'Model':<30} {'all 10 mean':<14} {'seed CI':<22} {'non-Kenya 6 mean':<18} {'seed CI':<22}")
    print('-' * 110)
    for m in models:
        vals_all = [pairs[k][m] for k in all_keys if m in pairs[k]]
        vals_nk = [pairs[k][m] for k in nonkenya if m in pairs[k]]
        if not vals_all:
            continue
        # Pooled mean and pooled SE assuming each transfer is one observation with its own (mean, std)
        mean_all = sum(v[0] for v in vals_all) / len(vals_all)
        # Use between-transfer std (sd of means) for cross-region uncertainty
        var_between = sum((v[0] - mean_all)**2 for v in vals_all) / max(len(vals_all)-1, 1)
        se_all = sqrt(var_between / len(vals_all))
        t_crit = 2.262 if len(vals_all) >= 10 else (2.571 if len(vals_all) >= 6 else T_CRIT_2DF_95)
        ci_all = (mean_all - t_crit*se_all, mean_all + t_crit*se_all)

        mean_nk = sum(v[0] for v in vals_nk) / len(vals_nk)
        var_nk = sum((v[0] - mean_nk)**2 for v in vals_nk) / max(len(vals_nk)-1, 1)
        se_nk = sqrt(var_nk / len(vals_nk))
        t_crit_nk = 2.571 if len(vals_nk) >= 6 else T_CRIT_2DF_95
        ci_nk = (mean_nk - t_crit_nk*se_nk, mean_nk + t_crit_nk*se_nk)

        print(f"{m:<30} {mean_all:.3f}         [{ci_all[0]:.3f},{ci_all[1]:.3f}]   {mean_nk:.3f}             [{ci_nk[0]:.3f},{ci_nk[1]:.3f}]")

    # Paired delta: frozen_prithvi vs unet across all 10 transfers
    print()
    print("## Paired delta: frozen_Prithvi - U-Net (per transfer)")
    deltas_all = []
    deltas_nk = []
    for k in all_keys:
        fp = pairs[k]['frozen_prithvi_cross'][0]
        un = pairs[k]['unet_cross'][0]
        deltas_all.append(fp - un)
        if 'kenya' not in k:
            deltas_nk.append(fp - un)
    mean_d_all = sum(deltas_all) / len(deltas_all)
    var_d_all = sum((d - mean_d_all)**2 for d in deltas_all) / max(len(deltas_all)-1, 1)
    se_d_all = sqrt(var_d_all / len(deltas_all))
    ci_d_all = (mean_d_all - 2.262*se_d_all, mean_d_all + 2.262*se_d_all)
    print(f"  All 10:    mean delta = {mean_d_all:+.4f}, 95% CI = [{ci_d_all[0]:+.4f}, {ci_d_all[1]:+.4f}], n={len(deltas_all)}")

    mean_d_nk = sum(deltas_nk) / len(deltas_nk)
    var_d_nk = sum((d - mean_d_nk)**2 for d in deltas_nk) / max(len(deltas_nk)-1, 1)
    se_d_nk = sqrt(var_d_nk / len(deltas_nk))
    ci_d_nk = (mean_d_nk - 2.571*se_d_nk, mean_d_nk + 2.571*se_d_nk)
    print(f"  Non-Kenya: mean delta = {mean_d_nk:+.4f}, 95% CI = [{ci_d_nk[0]:+.4f}, {ci_d_nk[1]:+.4f}], n={len(deltas_nk)}")

    # Equivalence: is frozen ~ unet within +-0.05 AUROC?
    EPS_XFER = 0.05
    eq_all = ci_d_all[0] > -EPS_XFER and ci_d_all[1] < EPS_XFER
    eq_nk = ci_d_nk[0] > -EPS_XFER and ci_d_nk[1] < EPS_XFER
    print(f"  TOST equivalence (eps=+-{EPS_XFER}): all10={eq_all}, nonKenya={eq_nk}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
