"""Canonical train/test split per country, replicating EXACTLY the filter+split of
scripts/ftw_controlled_label_comparison.py, so every baseline (U-Net, fine-tuned FMs,
low-data sweep) trains on the same train chips and is scored on the IDENTICAL test pixels
as the FM linear probes / RF. Verifies counts against the committed controlled JSON.

  python scripts/ftw_export_split.py --country india
Outputs: data/results/ftw_split_{C}.json (train chips + counts)
         data/results/ftw_split_{C}_test.parquet (test pixels: chip_id,pixel_r,pixel_c,label)
"""
import argparse, json, glob, numpy as np, pandas as pd, rasterio
from sklearn.model_selection import GroupShuffleSplit
ap = argparse.ArgumentParser(); ap.add_argument("--country", default="india"); ap.add_argument("--seed", type=int, default=20260514)
a = ap.parse_args(); C = a.country
D = f"data/features_per_pixel_ftw_{C}_true"; CHIPS = f"data/chips_ftw_{C}"
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
ok = ((proxy >= 0) & (~np.isnan(spec).any(axis=1)) & (~np.all(spec == 0, axis=1)))
ok = ok.to_numpy() if hasattr(ok, "to_numpy") else ok
m2 = m[ok].reset_index(drop=True); chips = m2.chip_id.to_numpy()
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=a.seed).split(np.zeros(len(m2)), m2.label, groups=chips))
train_chips = sorted(set(chips[tr])); test_chips = sorted(set(chips[te]))
ov = len(set(chips[tr]) & set(chips[te]))
test = m2.iloc[te][["chip_id", "pixel_r", "pixel_c", "label"]].reset_index(drop=True)
test.to_parquet(f"data/results/ftw_split_{C}_test.parquet")
out = {"country": C, "seed": a.seed, "n_pixels": int(len(m2)), "n_train_chips": len(train_chips),
       "n_test_chips": len(test_chips), "n_test_pixels": int(len(test)), "chip_overlap": int(ov),
       "train_chips": train_chips}
json.dump(out, open(f"data/results/ftw_split_{C}.json", "w"), indent=2)
# verify against committed controlled JSON
cj = f"data/results/ftw_controlled_label_comparison{'' if C=='india' else '_'+C}.json"
try:
    j = json.load(open(cj)); match = (j["n_test_chips"] == len(test_chips) and j["n_train_chips"] == len(train_chips) and j["n_pixels"] == len(m2))
    print("[%s] split: %d px, %d train / %d test chips, overlap=%d | controlled JSON match=%s (ctrl: %dpx %dtr %dte)"
          % (C, len(m2), len(train_chips), len(test_chips), ov, match, j["n_pixels"], j["n_train_chips"], j["n_test_chips"]))
except Exception as e:
    print("[%s] split: %d px, %d train / %d test chips, overlap=%d | (no controlled JSON to verify: %s)" % (C, len(m2), len(train_chips), len(test_chips), ov, e))
