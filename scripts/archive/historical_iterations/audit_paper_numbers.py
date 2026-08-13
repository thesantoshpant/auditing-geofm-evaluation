"""Cross-check paper-table numbers against shipped JSONs.

Prints every headline number alongside its source JSON so reviewers can diff
against the PDF tables directly.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "-"


def first(v):
    return v[0] if isinstance(v, list) else v


def main() -> int:
    print("=" * 70)
    print("Table 1 (Section 4): per-region in-region AUROC")
    print("Source: data/results/ftw_master_results.json")
    print("=" * 70)
    d = json.loads((RES / "ftw_master_results.json").read_text())
    print(f"{'region':12s}  {'RF_true':>7s}  {'RF_proxy':>8s}  {'U-Net':>7s}  {'Prithvi-ft':>10s}  {'frozenFM':>8s}")
    for c, v in d.items():
        print(f"{c:12s}  {fmt(v.get('rf_true')):>7s}  {fmt(v.get('rf_proxy')):>8s}  "
              f"{fmt(first(v.get('unet'))):>7s}  {fmt(first(v.get('prithvi_ft'))):>10s}  "
              f"{fmt(v.get('frozen_best_fm')):>8s}")

    print()
    print("=" * 70)
    print("Cross-region transfer (Table 2 RIGHT + Appendix transfer table)")
    print("Source: data/results/ftw_transfer_strengthened.json")
    print("=" * 70)
    t = json.loads((RES / "ftw_transfer_strengthened.json").read_text())
    pairs = t.get("pairs", t)
    keys = ("unet_cross", "prithvi_ft_cross", "frozen_prithvi_cross", "frozen_terramind_cross")
    means = {k: [] for k in keys}
    print(f"{'pair':30s}  " + "  ".join(f"{k.replace('_cross',''):>17s}" for k in keys))
    for pair, vals in pairs.items():
        row = []
        for k in keys:
            v = vals.get(k)
            row.append(fmt(first(v)))
            if isinstance(v, list):
                means[k].append(v[0])
        print(f"{pair:30s}  " + "  ".join(f"{r:>17s}" for r in row))
    print(f"{'MEAN':30s}  " + "  ".join(
        f"{sum(means[k])/len(means[k]):>17.3f}" if means[k] else f"{'-':>17s}" for k in keys))

    print()
    print("=" * 70)
    print("Appendix multi-metric (AUROC / AP / F1 / IoU, single seed)")
    print("=" * 70)
    by = (
        ("U-Net  ", RES / "ftw_unet_xmetrics.json",                "unet_true"),
        ("Frozen ", RES / "ftw_finetune_fm_prithvi_xm_frozen.json", "ft_true"),
        ("Full-FT", RES / "ftw_finetune_fm_prithvi_xm_ft.json",     "ft_true"),
    )
    for model, path, prefix in by:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        print(f"\n  {model}:")
        for c, v in d.items():
            print(f"    {c:12s}  AUROC={fmt(v.get(prefix + '_auroc'))}  "
                  f"AP={fmt(v.get(prefix + '_ap'))}  "
                  f"F1={fmt(v.get(prefix + '_f1'))}  "
                  f"IoU={fmt(v.get(prefix + '_iou'))}")

    print("\nDone. Reviewer: diff this output against the corresponding tables in the PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
