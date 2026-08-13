""": FTW partial-labeling sensitivity. FTW annotates only a SUBSET of fields per
chip, so some label=0 ("negative") pixels are actually unlabeled fields -- and the pixels
most likely to be hidden fields are those that look crop-like (WorldCover=cropland). We
therefore re-score the TRUE-label task with negatives RESTRICTED to high-confidence
non-cropland (label==0 AND WorldCover!=cropland), dropping ambiguous label==0 & crop-like
pixels. If the foundation-model >> spectral-RF ranking is stable under this restriction,
the conclusion is not an artifact of FTW partial labeling. Replicates the exact filter +
split of ftw_controlled_label_comparison.py.

  python scripts/ftw_partial_label_sensitivity.py            # india, france, kenya
"""
import sys, json, glob, numpy as np, pandas as pd, rasterio
sys.path.insert(0, "scripts")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
COUNTRIES = sys.argv[1:] if len(sys.argv) > 1 else ["india", "france", "kenya"]
SEED = 20260514
out = {}
for C in COUNTRIES:
    D = "data/features_per_pixel_ftw_%s_true" % C; CHIPS = "data/chips_ftw_%s" % C
    m = pd.read_parquet("%s/features_per_pixel_meta.parquet" % D).reset_index(drop=True)
    wcf, s2f = {}, {}
    for f in glob.glob("%s/*/*_worldcover.tif" % CHIPS): wcf["_".join(f.split("/")[-1].split("_")[:3])] = f
    for f in glob.glob("%s/*/*_s2.tif" % CHIPS): s2f[f.split("/")[-1].rsplit("_s2.tif", 1)[0]] = f
    proxy = np.full(len(m), -1, dtype=np.int8); nb = None; spec = None
    for cid, grp in m.groupby("chip_id"):
        if cid in wcf:
            with rasterio.open(wcf[cid]) as s: wc = s.read(1)
            for i, r in grp.iterrows():
                rr, cc = int(r.pixel_r), int(r.pixel_c)
                if 0 <= rr < wc.shape[0] and 0 <= cc < wc.shape[1]: proxy[i] = int(wc[rr, cc] == 40)
        if cid in s2f:
            with rasterio.open(s2f[cid]) as s: arr = s.read()
            if nb is None: nb = arr.shape[0]; spec = np.full((len(m), nb), np.nan, dtype=np.float32)
            for i, r in grp.iterrows():
                rr, cc = int(r.pixel_r), int(r.pixel_c)
                if 0 <= rr < arr.shape[1] and 0 <= cc < arr.shape[2]: spec[i] = arr[:, rr, cc]
    ok = ((proxy >= 0) & (~np.isnan(spec).any(axis=1)) & (~np.all(spec == 0, axis=1)))
    ok = ok.to_numpy() if hasattr(ok, "to_numpy") else ok
    m2 = m[ok].reset_index(drop=True); spec2 = spec[ok]; px = proxy[ok]; chips = m2.chip_id.to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(np.zeros(len(m2)), m2.label, groups=chips))
    yt = m2.label.to_numpy()
    # high-confidence test mask: keep positives + (negatives that are WC non-cropland)
    keep = (yt == 1) | (px == 0)
    te_full = te; te_hc = te[keep[te]]
    pf = glob.glob("%s/features_prithvi_eo_2_0_300m.npy" % D)
    Xp = np.load(pf[0])[ok] if pf else None
    def rf_auroc(idx):
        c = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=8, random_state=0).fit(spec2[tr], yt[tr])
        p = c.predict_proba(spec2[idx])[:, 1]; return round(roc_auc_score(yt[idx], p), 3)
    def lin_auroc(X, idx):
        v = ~np.all(X == 0, axis=1); tri = tr[v[tr]]; sc = StandardScaler().fit(X[tri])
        c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X[tri]), yt[tri])
        p = c.predict_proba(sc.transform(X[idx]))[:, 1]; return round(roc_auc_score(yt[idx], p), 3)
    rf_full = rf_auroc(te_full); rf_hc = rf_auroc(te_hc)
    pr_full = lin_auroc(Xp, te_full) if Xp is not None else None
    pr_hc = lin_auroc(Xp, te_hc) if Xp is not None else None
    n_drop = len(te_full) - len(te_hc)
    out[C] = {"n_test_full": int(len(te_full)), "n_test_highconf": int(len(te_hc)), "n_ambiguous_neg_dropped": int(n_drop),
              "rf_true_full": rf_full, "rf_true_highconf": rf_hc,
              "prithvi_true_full": pr_full, "prithvi_true_highconf": pr_hc,
              "fm_minus_rf_full": round((pr_full - rf_full), 3) if pr_full else None,
              "fm_minus_rf_highconf": round((pr_hc - rf_hc), 3) if pr_hc else None}
    print("%s: dropped %d ambiguous negs (%d->%d). RF %.3f->%.3f | Prithvi %.3f->%.3f | FM-RF %.3f->%.3f"
          % (C, n_drop, len(te_full), len(te_hc), rf_full, rf_hc, pr_full, pr_hc, out[C]["fm_minus_rf_full"], out[C]["fm_minus_rf_highconf"]), flush=True)
    json.dump(out, open("data/results/ftw_partial_label_sensitivity.json", "w"), indent=2)
print("wrote data/results/ftw_partial_label_sensitivity.json")
