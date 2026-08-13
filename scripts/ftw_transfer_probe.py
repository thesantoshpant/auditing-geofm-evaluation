"""Cross-region transfer via FROZEN FM features (+ spectral RF), multi-FM and multi-seed.
Train a probe on region A's TRUE-label train pixels, evaluate on region B's TRUE-label
test pixels. Reports in-region + cross-region AUROC mean+/-std over 3 split seeds, for
each FM and each directed pair. Replicates the controlled filter (proxy>=0 & valid spec).

  python scripts/ftw_transfer_probe.py            # all pairs, all FMs, 3 seeds
"""
import sys, glob, json, numpy as np, pandas as pd, rasterio, statistics as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
SEEDS = [20260514, 1, 2]
FMS = ["prithvi_eo_2_0_300m", "terramind_v1_base", "clay_v1"]
PAIRS = [("india", "cambodia"), ("cambodia", "india"), ("france", "netherlands"), ("netherlands", "france"),
         ("india", "france"), ("france", "india"), ("india", "kenya"), ("kenya", "india"),
         ("cambodia", "kenya"), ("kenya", "cambodia")]
REGIONS = sorted({c for p in PAIRS for c in p})
def load_meta(C):
    D = f"data/features_per_pixel_ftw_{C}_true"; CH = f"data/chips_ftw_{C}"
    m = pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
    wcf, s2f = {}, {}
    for f in glob.glob(f"{CH}/*/*_worldcover.tif"): wcf["_".join(f.split("/")[-1].split("_")[:3])] = f
    for f in glob.glob(f"{CH}/*/*_s2.tif"): s2f[f.split("/")[-1].rsplit("_s2.tif", 1)[0]] = f
    proxy = np.full(len(m), -1, np.int8); nb = None; spec = None
    for cid, grp in m.groupby("chip_id"):
        if cid in wcf:
            with rasterio.open(wcf[cid]) as s: wc = s.read(1)
            for i, r in grp.iterrows():
                rr, cc = int(r.pixel_r), int(r.pixel_c)
                if 0 <= rr < wc.shape[0] and 0 <= cc < wc.shape[1]: proxy[i] = int(wc[rr, cc] == 40)
        if cid in s2f:
            with rasterio.open(s2f[cid]) as s: arr = s.read()
            if nb is None: nb = arr.shape[0]; spec = np.full((len(m), nb), np.nan, np.float32)
            for i, r in grp.iterrows():
                rr, cc = int(r.pixel_r), int(r.pixel_c)
                if 0 <= rr < arr.shape[1] and 0 <= cc < arr.shape[2]: spec[i] = arr[:, rr, cc]
    ok = ((proxy >= 0) & (~np.isnan(spec).any(1)) & (~np.all(spec == 0, 1)))
    ok = ok.to_numpy() if hasattr(ok, "to_numpy") else ok
    m2 = m[ok].reset_index(drop=True)
    return {"ok": ok, "spec": spec[ok], "y": m2.label.to_numpy(), "chips": m2.chip_id.to_numpy()}
def split(meta, seed):
    return next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(np.zeros(len(meta["y"])), meta["y"], groups=meta["chips"]))
def auroc(y, p): return roc_auc_score(y, p)
META = {C: load_meta(C) for C in REGIONS}
SPLITS = {C: {s: split(META[C], s) for s in SEEDS} for C in REGIONS}
def feat(C, fm): return np.load(f"data/features_per_pixel_ftw_{C}_true/features_{fm}.npy")[META[C]["ok"]]
out = {}
# spectral RF (FM-independent) per pair/seed
for A, B in PAIRS:
    key = f"{A}->{B}"; out[key] = {}
    rf_cross = []
    for s in SEEDS:
        trA, _ = SPLITS[A][s]; _, teB = SPLITS[B][s]
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=8, random_state=0).fit(META[A]["spec"][trA], META[A]["y"][trA])
        rf_cross.append(auroc(META[B]["y"][teB], rf.predict_proba(META[B]["spec"][teB])[:, 1]))
    out[key]["spectral_rf"] = {"cross_mean": round(st.mean(rf_cross), 4), "cross_std": round(st.pstdev(rf_cross), 4)}
    print(f"{key} spectral_rf cross={round(st.mean(rf_cross),4)}+/-{round(st.pstdev(rf_cross),4)}", flush=True)
    json.dump(out, open("data/results/ftw_transfer_probe_multi.json", "w"), indent=2)
for fm in FMS:
    fc = {C: feat(C, fm) for C in REGIONS}
    for A, B in PAIRS:
        key = f"{A}->{B}"; inr, cross = [], []
        for s in SEEDS:
            trA, teA = SPLITS[A][s]; _, teB = SPLITS[B][s]
            sc = StandardScaler().fit(fc[A][trA]); clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(fc[A][trA]), META[A]["y"][trA])
            inr.append(auroc(META[A]["y"][teA], clf.predict_proba(sc.transform(fc[A][teA]))[:, 1]))
            cross.append(auroc(META[B]["y"][teB], clf.predict_proba(sc.transform(fc[B][teB]))[:, 1]))
        out[key][fm] = {"in_region_mean": round(st.mean(inr), 4), "in_region_std": round(st.pstdev(inr), 4),
                        "cross_mean": round(st.mean(cross), 4), "cross_std": round(st.pstdev(cross), 4),
                        "transfer_gap": round(st.mean(inr) - st.mean(cross), 4)}
        print(f"{key} {fm}: in={round(st.mean(inr),4)} cross={round(st.mean(cross),4)}+/-{round(st.pstdev(cross),4)} gap={round(st.mean(inr)-st.mean(cross),4)}", flush=True)
        json.dump(out, open("data/results/ftw_transfer_probe_multi.json", "w"), indent=2)
    del fc
print("wrote data/results/ftw_transfer_probe_multi.json")
