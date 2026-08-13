"""Strictly-controlled label comparison on IDENTICAL pixels+features, for any
FTW country. Hold features fixed, vary ONLY the target: FTW true field-membership
vs WorldCover-cropland proxy. Models: every FM (incl. AnySat from its aligned
dir) + RF on raw S2 reflectance. Records chip-overlap for provenance.

    python scripts/ftw_controlled_label_comparison.py --country india   # default
    python scripts/ftw_controlled_label_comparison.py --country kenya
"""
import argparse, json, glob, numpy as np, pandas as pd, rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--country", default="india")
ap.add_argument("--seed", type=int, default=20260514)
ap.add_argument("--group-by", choices=["chip","tile"], default="chip",
                help="split grouping: chip_id (default) or MGRS tile (scene-disjoint robustness)")
args = ap.parse_args()
C = args.country
D = f"data/features_per_pixel_ftw_{C}_true"
ANY = f"data/features_per_pixel_ftw_{C}_true_anysat"
CHIPS = f"data/chips_ftw_{C}"
OUT = f"data/results/ftw_controlled_label_comparison_{C}.json" if C != "india" else "data/results/ftw_controlled_label_comparison.json"

m = pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
wcf, s2f = {}, {}
for f in glob.glob(f"{CHIPS}/*/*_worldcover.tif"): wcf["_".join(f.split("/")[-1].split("_")[:3])] = f
for f in glob.glob(f"{CHIPS}/*/*_s2.tif"): s2f[f.split("/")[-1].rsplit("_s2.tif", 1)[0]] = f
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
m = m.assign(proxy_label=proxy)
ok = ((proxy >= 0) & (~np.isnan(spec).any(axis=1)) & (~np.all(spec == 0, axis=1)))
ok = ok.to_numpy() if hasattr(ok, "to_numpy") else ok
m2 = m[ok].reset_index(drop=True); spec = spec[ok]; chips = m2.chip_id.to_numpy()
if args.group_by == "tile":
    import json as _json
    c2tile = {}
    for _ln in open(f"data/chips/manifest_ftw_{C}.jsonl"):
        _r = _json.loads(_ln); c2tile[_r["chip_id"]] = str(_r.get("scene_id","")).split("_")[-1]
    groups = np.array([c2tile.get(c, c) for c in chips])
    OUT = OUT.replace(".json", "_tilesplit.json")
    print("[%s] GROUP-BY TILE: %d distinct tiles" % (C, len(set(groups))))
else:
    groups = chips
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed).split(np.zeros(len(m2)), m2.label, groups=groups))
ov = len(set(groups[tr]) & set(groups[te]))
print("[%s] CONTROLLED: %d pixels (nodata excl), %d train / %d test chips, OVERLAP=%d" % (C, len(m2), len(set(chips[tr])), len(set(chips[te])), ov))
print("true pos_rate=%.3f  proxy pos_rate=%.3f  bands=%d\n" % (m2.label.mean(), m2.proxy_label.mean(), nb))

def lin(X, y):
    v = ~np.all(X == 0, axis=1); tri, tei = tr[v[tr]], te[v[te]]
    if len(set(y[tei])) < 2: return None, None
    sc = StandardScaler().fit(X[tri]); c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X[tri]), y[tri])
    p = c.predict_proba(sc.transform(X[tei]))[:, 1]; return round(roc_auc_score(y[tei], p), 3), round(f1_score(y[tei], (p >= .5).astype(int)), 3)

def rf(X, y):
    if len(set(y[te])) < 2: return None, None
    c = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=8, random_state=0).fit(X[tr], y[tr])
    p = c.predict_proba(X[te])[:, 1]; return round(roc_auc_score(y[te], p), 3), round(f1_score(y[te], (p >= .5).astype(int)), 3)

res = {"country": C, "n_pixels": int(len(m2)), "chip_overlap": int(ov), "n_train_chips": int(len(set(chips[tr]))),
       "n_test_chips": int(len(set(chips[te]))), "true_pos_rate": round(float(m2.label.mean()), 3),
       "proxy_pos_rate": round(float(m2.proxy_label.mean()), 3), "seed": args.seed, "models": {}}
yt, yp = m2.label.to_numpy(), m2.proxy_label.to_numpy()
print("%-22s | TRUE field        | WC-cropland proxy" % "model (AUROC / F1)"); print("-" * 72)
feats = sorted(glob.glob(f"{D}/features_*.npy")) + glob.glob(f"{ANY}/features_*.npy")
for f in feats:
    fm = f.split("/")[-1][len("features_"):-4]; X = np.load(f)[ok]
    at, ft = lin(X, yt); ap_, fp = lin(X, yp); res["models"][fm] = {"true": {"auroc": at, "f1": ft}, "proxy": {"auroc": ap_, "f1": fp}}
    print("%-22s | %s / %s      | %s / %s" % (fm, at, ft, ap_, fp))
art, frt = rf(spec, yt); arp, frp = rf(spec, yp); res["models"]["rf_spectral"] = {"true": {"auroc": art, "f1": frt}, "proxy": {"auroc": arp, "f1": frp}}
print("%-22s | %s / %s      | %s / %s" % ("rf_spectral(S2)", art, frt, arp, frp))
json.dump(res, open(OUT, "w"), indent=2)
print("\nwrote " + OUT)
