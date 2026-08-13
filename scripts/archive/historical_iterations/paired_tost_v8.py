"""v8 cohort paired-TOST analysis.

Separate seven-seed robustness cohort using new random seeds 3..9 from a single
batch of 168 runs (one environment, single recipe -- no cross-recipe mixing).
Reports a clean n=7 paired TOST plus per-cell TOST p-values and Holm familywise
adjusted p-values across the 12 region x model cells.

Writes ftw_paired_tost_v8.json. A post-hoc robustness companion to the n=3
originally specified analysis in the main text.
"""
import json
import os
import statistics
from math import sqrt

try:
    from scipy import stats as _scistats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]
EPS = 0.02
T_TABLE = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
V8_SEEDS = list(range(3, 10))


def t_cdf(x, df):
    """One-sided Student-t CDF. Uses scipy if available, else a numeric fallback."""
    if HAS_SCIPY:
        return float(_scistats.t.cdf(x, df))
    # Fallback: regularized incomplete beta. Acceptable accuracy for our use.
    # P(T <= x) = 1 - 0.5 * I_{df/(df+x^2)}(df/2, 1/2)  for x > 0
    #          =     0.5 * I_{df/(df+x^2)}(df/2, 1/2)  for x < 0
    from math import lgamma, log, exp
    def _betainc(a, b, z):
        # series for regularized incomplete beta via continued fraction (Lentz)
        if z == 0: return 0.0
        if z == 1: return 1.0
        lnpre = (a * log(z) + b * log(1 - z)
                 - log(a) - (lgamma(a) + lgamma(b) - lgamma(a + b)))
        # continued fraction expansion for I_z(a,b)
        EPS_ = 1e-12
        FPMIN = 1e-30
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1.0 - qab * z / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, 250):
            m2 = 2 * m
            aa = m * (b - m) * z / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * z / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS_:
                break
        return exp(lnpre) * h
    z = df / (df + x * x)
    bi = _betainc(df / 2.0, 0.5, z)
    return 1.0 - 0.5 * bi if x > 0 else 0.5 * bi


def tost_pvalue(mean_d, sd, n, eps):
    """Two one-sided t-tests; TOST p = max of the two one-sided p-values."""
    if sd == 0:
        return 0.0 if abs(mean_d) < eps else 1.0
    se = sd / sqrt(n)
    df = n - 1
    t_lower = (mean_d - (-eps)) / se   # tests H0: mean_d <= -eps
    t_upper = ((+eps) - mean_d) / se   # tests H0: mean_d >= +eps
    p_lower = 1.0 - t_cdf(t_lower, df)
    p_upper = 1.0 - t_cdf(t_upper, df)
    return max(p_lower, p_upper)


def holm_adjust(pvals):
    """Holm familywise correction. Returns list of adjusted p-values, same order."""
    k = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * k
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        a = min(1.0, (k - rank) * p)
        running_max = max(running_max, a)
        adj[orig_idx] = running_max
    return adj


def read(fname, region):
    p = os.path.join(R, fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get(region, {}).get("ft_true_auroc")


def v8(model, region, mode, seed):
    mode_word = "backbone" if mode == "frozen" else "none"
    pre = "prithviprithvi" if model == "prithvi" else "terramindterramind"
    return read(f"ftw_finetune_fm_{pre}_v8_{region}_{mode_word}_s{seed}.json", region)


def analyze(model, seeds):
    n = len(seeds)
    t_crit = T_TABLE.get(n - 1, 1.96)
    print(f"\n--- {model.title()} (separate seven-seed robustness cohort; seeds {seeds}; df={n-1}) ---")
    out = {}
    eq = 0
    missing = []
    for r in REGIONS:
        fb = [v8(model, r, "frozen", s) for s in seeds]
        ft = [v8(model, r, "ft", s) for s in seeds]
        if None in fb or None in ft:
            missing.append(f"{model}/{r}")
            continue
        diffs = [fb[i] - ft[i] for i in range(n)]
        m = statistics.mean(diffs)
        s = statistics.stdev(diffs) if n > 1 else 0.0
        se = s / sqrt(n)
        half = t_crit * se
        lo, hi = m - half, m + half
        tost = lo > -EPS and hi < EPS
        verdict = "EQUIV" if tost else "not-eq"
        print(f"  {r:<13} delta={m:+.4f}+-{s:.4f}  CI=[{lo:+.4f},{hi:+.4f}]  {verdict}")
        p_tost = tost_pvalue(m, s, n, EPS)
        out[r] = {
            "frozen_seeds": fb, "ft_seeds": ft,
            "paired_delta_mean": round(m, 4),
            "paired_delta_std": round(s, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "tost_equivalent": tost,
            "tost_p_value": p_tost,             # full precision; used for Holm
            "tost_p_value_display": round(p_tost, 6),
        }
        if tost:
            eq += 1
    print(f"  {model.title()}: {eq} of {len(out)} TOST-equivalent")
    return out, eq, missing


def main():
    import sys
    pri, pri_eq, pri_missing = analyze("prithvi", V8_SEEDS)
    tm, tm_eq, tm_missing = analyze("terramind", V8_SEEDS)
    missing = pri_missing + tm_missing
    if missing:
        sys.stderr.write(f"\nERROR: incomplete v8 grid; missing cells: {missing}\n")
        sys.exit(2)
    total = pri_eq + tm_eq
    print(f"\n*** v8 n=7: {total} of 12 region x model cells TOST-equivalent ***")

    # Holm familywise correction across all 12 TOST p-values.
    ordered_cells = []
    pvals = []
    for model_name, mout in (("prithvi", pri), ("terramind", tm)):
        for r in REGIONS:
            ordered_cells.append((model_name, r))
            pvals.append(mout[r]["tost_p_value"])
    adj = holm_adjust(pvals)                  # uses full-precision input
    for (model_name, r), a in zip(ordered_cells, adj):
        target = pri if model_name == "prithvi" else tm
        target[r]["tost_p_value_holm"] = a    # full precision
        target[r]["tost_p_value_holm_display"] = round(a, 6)
        target[r]["tost_equiv_holm_0_05"] = a < 0.05
        target[r]["tost_equiv_holm_0_025"] = a < 0.025

    n_holm_05 = sum(1 for c in adj if c < 0.05)
    n_holm_025 = sum(1 for c in adj if c < 0.025)
    print(f"\nHolm familywise correction across the 12 cells:")
    print(f"  at alpha=0.05 : {n_holm_05} of 12 cells retain TOST equivalence")
    print(f"  at alpha=0.025: {n_holm_025} of 12 cells retain TOST equivalence")

    out_path = os.path.join(R, "ftw_paired_tost_v8.json")
    with open(out_path, "w") as f:
        json.dump({
            "epsilon_auroc": EPS,
            "n_seeds": 7,
            "seeds": V8_SEEDS,
            "method": ("seed-paired t-interval (df=6); separate seven-seed robustness "
                       "cohort, single-recipe v8 batch (one compute environment, one "
                       "terratorch/timm pin, 168 runs). Reports per-cell TOST p-value "
                       "(max of two one-sided t-tests at margin epsilon=0.02) and Holm "
                       "familywise adjusted p-values across the 12 region x model cells."),
            "holm": {
                "n_cells": 12,
                "n_retained_alpha_0_05": n_holm_05,
                "n_retained_alpha_0_025": n_holm_025,
            },
            "prithvi": {"regions": pri, "n_equivalent": pri_eq},
            "terramind": {"regions": tm, "n_equivalent": tm_eq},
            "n_equivalent_combined": total,
            "n_total_cells": 12,
        }, f, indent=2)
    print(f"\nWrote data/results/{os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
