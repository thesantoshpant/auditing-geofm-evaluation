"""Train the supervised U-Net baseline used by the FTW evaluation.

``--epochs`` supports the 80-epoch budget control as well as the primary
150-epoch run. ``--eval-country`` evaluates a source-trained model on another
region's canonical test pixels.
"""
import sys, json, glob, argparse, numpy as np, rasterio, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import torch, torch.nn as nn, torch.nn.functional as F
from extract_features_per_pixel import build_polygon_id_raster
import geopandas as gpd
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, roc_auc_score
ap = argparse.ArgumentParser()
ap.add_argument("--robust", action="store_true"); ap.add_argument("--perpixel", action="store_true")
ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--tag", default=""); ap.add_argument("--eval-country", default=""); ap.add_argument("countries", nargs="*")
a = ap.parse_args()
COUNTRIES = a.countries if a.countries else ["india", "vietnam", "france", "netherlands", "kenya"]
dev = "cuda" if torch.cuda.is_available() else "cpu"; EP = a.epochs
def conv(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s):
        super().__init__(); s.e1 = conv(12, 64); s.e2 = conv(64, 128); s.e3 = conv(128, 256); s.e4 = conv(256, 512); s.b = conv(512, 1024); s.p = nn.MaxPool2d(2)
        s.u4 = nn.ConvTranspose2d(1024, 512, 2, 2); s.d4 = conv(1024, 512); s.u3 = nn.ConvTranspose2d(512, 256, 2, 2); s.d3 = conv(512, 256)
        s.u2 = nn.ConvTranspose2d(256, 128, 2, 2); s.d2 = conv(256, 128); s.u1 = nn.ConvTranspose2d(128, 64, 2, 2); s.d1 = conv(128, 64); s.o = nn.Conv2d(64, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.p(e1)); e3 = s.e3(s.p(e2)); e4 = s.e4(s.p(e3)); b = s.b(s.p(e4))
        d4 = s.d4(torch.cat([s.u4(b), e4], 1)); d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
        d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1)); return s.o(d1)[:, 0]
class PerPixel(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Conv2d(12, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 1, 1))
    def forward(s, x): return s.net(x)[:, 0]
def dice_loss(logit, y):
    p = torch.sigmoid(logit); return 1.0 - (2 * (p * y).sum() + 1.0) / ((p + y).sum() + 1.0)
def pad16(arr):
    _, H, W = arr.shape; ph, pw = (-H) % 16, (-W) % 16
    return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="reflect") if (ph or pw) else arr
def chips_of(C): return {f.split("/")[-1].rsplit("_s2.tif", 1)[0]: f for f in glob.glob(f"data/chips_ftw_{C}/*/*_s2.tif")}
def run(C):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    sp = json.load(open(f"data/results/ftw_split_{C}.json")); train_chips = sp["train_chips"]
    polys = gpd.read_parquet(f"data/labels/polygons_ftw_{C}_keyed.parquet"); pby = {}
    for _, p in polys.iterrows(): pby.setdefault(p["chip_id"], []).append({"polygon_id": p["polygon_id"], "district": p.get("district", ""), "geometry": p["geometry"], "area_m2": float(p["area_m2"]), "size_bin": p["size_bin"]})
    s2f = chips_of(C); train_chips = [c for c in train_chips if c in s2f]
    if a.frac < 1.0:
        r = np.random.RandomState(a.seed); r.shuffle(train_chips); train_chips = train_chips[:max(2, int(round(len(train_chips) * a.frac)))]
    raw = {}
    for c in train_chips:
        with rasterio.open(s2f[c]) as s: raw[c] = s.read().astype(np.float32) / 10000.0
    stk = np.concatenate([r.reshape(12, -1) for r in raw.values()], 1); MEAN = stk.mean(1).astype(np.float32); STD = (stk.std(1) + 1e-6).astype(np.float32)
    def mask(cid): pid, _ = build_polygon_id_raster(s2f[cid], pby.get(cid, [])); return (pid > 0).astype(np.float32)
    cache = {c: ((raw[c] - MEAN[:, None, None]) / STD[:, None, None], mask(c)) for c in train_chips}
    cov = float(np.mean([cache[c][1].mean() for c in train_chips]))
    pwv = min((1 - cov) / max(cov, 1e-3), 50.0) if a.robust else (1 - cov) / max(cov, 1e-3)
    net = (PerPixel() if a.perpixel else UNet()).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    if a.robust:
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5), torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(EP - 5, 1))], milestones=[5])
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pwv], device=dev)); tag = ("PerPixel" if a.perpixel else ("UNet-robust" if a.robust else "UNet"))
    ec = a.eval_country if a.eval_country else C
    print(f"{C}->{ec} [{tag} frac={a.frac:.2f} seed={a.seed}]: {len(train_chips)} train chips, cov={cov:.3f}", flush=True)
    net.train()
    for ep in range(EP):
        np.random.shuffle(train_chips);
        for i in range(0, len(train_chips), 4):
            xb = [pad16(cache[c][0]) for c in train_chips[i:i + 4]]; yb = [pad16(cache[c][1][None])[0] for c in train_chips[i:i + 4]]
            Hm = min(x.shape[1] for x in xb); Wm = min(x.shape[2] for x in xb)
            X = torch.tensor(np.stack([x[:, :Hm, :Wm] for x in xb])).to(dev); Y = torch.tensor(np.stack([y[:Hm, :Wm] for y in yb])).to(dev)
            if np.random.rand() < .5: X = torch.flip(X, [2]); Y = torch.flip(Y, [1])
            if np.random.rand() < .5: X = torch.flip(X, [3]); Y = torch.flip(Y, [2])
            opt.zero_grad(); o = net(X); l = bce(o, Y) + (dice_loss(o, Y) if a.robust else 0.0); l.backward(); opt.step()
        sch.step()
    # evaluate on ec's canonical test pixels (cross-region if ec != C)
    sp_e = json.load(open(f"data/results/ftw_split_{ec}.json")); test = pd.read_parquet(f"data/results/ftw_split_{ec}_test.parquet").reset_index(drop=True); s2e = chips_of(ec)
    net.eval(); probs = np.full(len(test), np.nan)
    with torch.no_grad():
        for cid, grp in test.groupby("chip_id"):
            if cid not in s2e: continue
            with rasterio.open(s2e[cid]) as s: a0 = s.read().astype(np.float32) / 10000.0
            H0, W0 = a0.shape[1], a0.shape[2]; arr = (pad16(a0) - MEAN[:, None, None]) / STD[:, None, None]
            pr = torch.sigmoid(net(torch.tensor(arr[None]).to(dev)))[0].cpu().numpy()
            for j, r in grp.iterrows():
                rr, cc = int(r.pixel_r), int(r.pixel_c)
                if rr < H0 and cc < W0: probs[j] = pr[rr, cc]
    nan = int(np.isnan(probs).sum()); assert nan == 0, f"{nan} unpredicted"
    y = test.label.to_numpy(); pred = (probs >= .5).astype(int)
    auc = round(roc_auc_score(y, probs), 4)
    avg_precision = round(average_precision_score(y, probs), 4)
    f1 = round(f1_score(y, pred), 4)
    iou = round(jaccard_score(y, pred, zero_division=0), 4)
    print(f"{C}->{ec} [{tag} seed={a.seed}] AUROC={auc} AP={avg_precision} F1={f1} IoU={iou} n={len(test)}\n", flush=True)
    return {"train_country": C, "eval_country": ec, "unet_true_auroc": auc, "unet_true_ap": avg_precision, "unet_true_f1": f1, "unet_true_iou": iou, "n_eval": int(len(test)), "frac": a.frac, "seed": a.seed, "epochs": EP}
OUTF = f"data/results/ftw_unet{a.tag}.json" if a.tag else ("data/results/ftw_unet_perpixel_ablation.json" if a.perpixel else ("data/results/ftw_unet_baseline_robust.json" if a.robust else "data/results/ftw_unet_baseline.json"))
out = {}
for C in COUNTRIES: out[C] = run(C); json.dump(out, open(OUTF, "w"), indent=2)
print("wrote " + OUTF)
