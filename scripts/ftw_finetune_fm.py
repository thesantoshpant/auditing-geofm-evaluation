"""Train a GeoFM decoder with a frozen or updated backbone.

The default 80-epoch recipe is shared by the frozen-decoder and full-fine-tune
conditions. ``--curve-every N`` records post-hoc test metrics every N epochs;
these trajectories are diagnostics and are not used for checkpoint selection.
``--eval-all-regions`` writes the six-target schema used by the headline grid.
"""
import sys, json, glob, os, argparse, numpy as np, rasterio, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import torch, torch.nn as nn, torch.nn.functional as F
from extract_features_per_pixel import build_polygon_id_raster, chip_from_geotiff_array
from ftw_eval.evaluation import REGIONS, evaluation_key, evaluation_targets
import geopandas as gpd
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["prithvi", "terramind"], default="prithvi")
ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--tag", default="")
ap.add_argument("--output", default="")
ap.add_argument("--curve-every", type=int, default=0)
ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--freeze", choices=["none", "backbone", "last6"], default="none")
ap.add_argument("--eval-country", default=""); ap.add_argument("--eval-all-regions", action="store_true")
ap.add_argument("countries", nargs="*")
a = ap.parse_args()
if a.eval_all_regions and a.eval_country:
    ap.error("--eval-all-regions and --eval-country are mutually exclusive")
if a.eval_all_regions and a.curve_every:
    ap.error("--eval-all-regions cannot be combined with --curve-every")
COUNTRIES = a.countries if a.countries else list(REGIONS)
dev = "cuda"
def make_fm():
    if a.model == "prithvi": from ftw_eval.model_zoo.prithvi import PrithviFoundationModel as M
    else: from ftw_eval.model_zoo.terramind import TerraMindFoundationModel as M
    fm = M(device=dev); fm.load(); return fm
def fwd_tokens(fm, x):
    if a.model == "prithvi":
        out = fm._model.forward_features(x)
        if isinstance(out, (list, tuple)): out = out[-1]
        if out.dim() == 4: out = out.flatten(2).transpose(1, 2)
        tok = out[:, 1:]
    else:
        out = fm._model({"S2L2A": x}); tok = out[-1] if isinstance(out, (list, tuple)) else out
        if tok.dim() == 4: tok = tok.flatten(2).transpose(1, 2)
    B, N, D = tok.shape; h = w = int(round(N ** 0.5)); return tok.transpose(1, 2).reshape(B, D, h, w)
class Decoder(nn.Module):
    def __init__(s, D):
        super().__init__(); ch = [D, 256, 128, 64, 32]; L = []
        for i in range(4): L += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(True)]
        s.body = nn.Sequential(*L); s.head = nn.Conv2d(32, 1, 1)
    def forward(s, x): return s.head(s.body(x))[:, 0]
def prep(fm, chip):
    d = fm.preprocess(chip); t = d["pixel_values"] if a.model == "prithvi" else d["S2L2A"]; return t.squeeze(0).cpu()
def dice_loss(logit, y):
    p = torch.sigmoid(logit); return 1.0 - (2 * (p * y).sum() + 1.0) / ((p + y).sum() + 1.0)
def set_freeze(fm):
    if a.freeze == "none":
        for p in fm._model.parameters(): p.requires_grad_(True)
    else:
        for p in fm._model.parameters(): p.requires_grad_(False)
        if a.freeze == "last6":
            blocks = getattr(fm._model, "blocks", None)
            if blocks is not None:
                for blk in list(blocks)[-6:]:
                    for p in blk.parameters(): p.requires_grad_(True)
def masks_for(C):
    polys = gpd.read_parquet(f"data/labels/polygons_ftw_{C}_keyed.parquet"); pby = {}
    for _, p in polys.iterrows(): pby.setdefault(p["chip_id"], []).append({"polygon_id": p["polygon_id"], "district": p.get("district", ""), "geometry": p["geometry"], "area_m2": float(p["area_m2"]), "size_bin": p["size_bin"]})
    return pby
def chips_of(C): return {os.path.basename(f).rsplit("_s2.tif", 1)[0]: f for f in glob.glob(f"data/chips_ftw_{C}/*/*_s2.tif")}
def run(fm, C):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    sp = json.load(open(f"data/results/ftw_split_{C}.json")); train_chips = sp["train_chips"]
    pby = masks_for(C); s2f = chips_of(C); train_chips = [c for c in train_chips if c in s2f]
    if a.frac < 1.0:
        r = np.random.RandomState(a.seed); r.shuffle(train_chips); train_chips = train_chips[:max(2, int(round(len(train_chips) * a.frac)))]
    Xc, Yc = [], []
    for c in train_chips:
        chip, _, _ = chip_from_geotiff_array(s2f[c]); Xc.append(prep(fm, chip))
        pid, _ = build_polygon_id_raster(s2f[c], pby.get(c, [])); m = (pid > 0).astype(np.float32)
        Yc.append(F.interpolate(torch.tensor(m)[None, None], size=(224, 224), mode="nearest")[0, 0])
    cov = float(np.mean([y.mean().item() for y in Yc])); pw = torch.tensor([min((1 - cov) / max(cov, 1e-3), 50.0)], device=dev)
    D = fwd_tokens(fm, Xc[0][None].to(dev)).shape[1]; dec = Decoder(D).to(dev); set_freeze(fm); fm._model.train(); dec.train()
    opt = torch.optim.AdamW([{"params": [p for p in fm._model.parameters() if p.requires_grad], "lr": 1e-4}, {"params": dec.parameters(), "lr": 1e-3}])
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5), torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.epochs - 5, 1))], milestones=[5])
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    eval_regions = evaluation_targets(C, a.eval_country, a.eval_all_regions)
    diagnostic_region = a.eval_country or C
    print(f"{C} [{a.model}-ft frac={a.frac:.2f} seed={a.seed} freeze={a.freeze}]: {len(train_chips)} chips cov={cov:.3f}", flush=True)
    def evaluate(ec):
        test = pd.read_parquet(f"data/results/ftw_split_{ec}_test.parquet").reset_index(drop=True); s2e = chips_of(ec)
        fm._model.eval(); dec.eval(); probs = np.full(len(test), np.nan)
        with torch.no_grad():
            for cid, grp in test.groupby("chip_id"):
                if cid not in s2e: continue
                chip, H0, W0 = chip_from_geotiff_array(s2e[cid]); x = prep(fm, chip)[None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = dec(fwd_tokens(fm, x))
                pr = F.interpolate(torch.sigmoid(lg.float())[None], size=(H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                for j, r in grp.iterrows():
                    rr, cc = int(r.pixel_r), int(r.pixel_c)
                    if rr < H0 and cc < W0: probs[j] = pr[rr, cc]
        nan = int(np.isnan(probs).sum()); assert nan == 0, f"{nan} unpredicted"
        y = test.label.to_numpy(); pred = (probs >= .5).astype(int)
        return {
            "auroc": round(roc_auc_score(y, probs), 4),
            "ap": round(average_precision_score(y, probs), 4),
            "f1": round(f1_score(y, pred), 4),
            "iou": round(jaccard_score(y, pred, zero_division=0), 4),
            "n_eval": int(len(test)),
        }

    order = list(range(len(train_chips))); epoch_curve = []
    for ep in range(a.epochs):
        np.random.shuffle(order)
        for i in range(0, len(order), 4):
            idx = order[i:i + 4]; xb = torch.stack([Xc[k] for k in idx]).to(dev); yb = torch.stack([Yc[k] for k in idx]).to(dev)
            if np.random.rand() < .5: xb = torch.flip(xb, [xb.dim() - 1]); yb = torch.flip(yb, [2])
            if np.random.rand() < .5: xb = torch.flip(xb, [xb.dim() - 2]); yb = torch.flip(yb, [1])
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = dec(fwd_tokens(fm, xb)).float(); l = bce(logit, yb) + dice_loss(logit, yb)
            l.backward(); opt.step()
        sch.step()
        if a.curve_every and (ep + 1) % a.curve_every == 0:
            metrics = evaluate(diagnostic_region); epoch_curve.append({"epoch": ep + 1, **metrics})
            fm._model.train(); dec.train()
    results = {}
    for ec in eval_regions:
        metrics = epoch_curve[-1].copy() if epoch_curve and ec == diagnostic_region and epoch_curve[-1]["epoch"] == a.epochs else evaluate(ec)
        metrics.pop("epoch", None)
        n_eval = metrics.pop("n_eval")
        print(f"{C}->{ec} [{a.model}-ft seed={a.seed} freeze={a.freeze}] AUROC={metrics['auroc']} AP={metrics['ap']} F1={metrics['f1']} IoU={metrics['iou']} n={n_eval}", flush=True)
        result = {"train_country": C, "eval_country": ec, "ft_true_auroc": metrics["auroc"], "ft_true_ap": metrics["ap"], "ft_true_f1": metrics["f1"], "ft_true_iou": metrics["iou"], "n_eval": n_eval, "frac": a.frac, "seed": a.seed, "freeze": a.freeze}
        if not a.eval_all_regions:
            result["epochs"] = a.epochs
        if epoch_curve:
            result["epoch_curve"] = epoch_curve
            result["log_every"] = a.curve_every
        key = evaluation_key(C, ec)
        results[key] = result
    print(flush=True)
    return results
OUTF = a.output or ("data/results/ftw_finetune_fm_" + a.model + a.tag + ".json")
os.makedirs(os.path.dirname(os.path.abspath(OUTF)), exist_ok=True)
out = {}
for C in COUNTRIES:
    out.update(run(make_fm(), C))
    with open(OUTF, "w") as handle: json.dump(out, handle, indent=2)
print("wrote " + OUTF)
