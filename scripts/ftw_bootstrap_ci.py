"""Chip-clustered bootstrap 95% CIs for the controlled label comparison.
Train each model once on the train split; then bootstrap-resample TEST CHIPS
(with replacement, B=1000) to get CIs on each model's true & proxy AUROC and on
the key PAIRED differences (uses the same chip resamples, so the paired CI is
honest). Reuses cached TRUE-mode features; no GPU.
    python ftw_bootstrap_ci.py --country india|kenya
"""
import argparse, glob, json, numpy as np, pandas as pd, rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

ap=argparse.ArgumentParser(); ap.add_argument("--country",default="india"); ap.add_argument("--B",type=int,default=1000); ap.add_argument("--seed",type=int,default=20260514)
a=ap.parse_args(); C=a.country
D=f"data/features_per_pixel_ftw_{C}_true"; ANY=f"data/features_per_pixel_ftw_{C}_true_anysat"; CH=f"data/chips_ftw_{C}"
m=pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
wcf={ "_".join(f.split("/")[-1].split("_")[:3]):f for f in glob.glob(f"{CH}/*/*_worldcover.tif")}
s2f={ f.split("/")[-1].rsplit("_s2.tif",1)[0]:f for f in glob.glob(f"{CH}/*/*_s2.tif")}
proxy=np.full(len(m),-1,np.int8); nb=None; spec=None
for cid,grp in m.groupby("chip_id"):
    if cid in wcf:
        with rasterio.open(wcf[cid]) as s: wc=s.read(1)
        for i,r in grp.iterrows():
            rr,cc=int(r.pixel_r),int(r.pixel_c)
            if 0<=rr<wc.shape[0] and 0<=cc<wc.shape[1]: proxy[i]=int(wc[rr,cc]==40)
    if cid in s2f:
        with rasterio.open(s2f[cid]) as s: arr=s.read()
        if nb is None: nb=arr.shape[0]; spec=np.full((len(m),nb),np.nan,np.float32)
        for i,r in grp.iterrows():
            rr,cc=int(r.pixel_r),int(r.pixel_c)
            if 0<=rr<arr.shape[1] and 0<=cc<arr.shape[2]: spec[i]=arr[:,rr,cc]
ok=((proxy>=0)&(~np.isnan(spec).any(1))&(~np.all(spec==0,1)))
ok=ok.to_numpy() if hasattr(ok,"to_numpy") else ok
m=m[ok].reset_index(drop=True); spec=spec[ok]; proxy=proxy[ok]
yt=m.label.to_numpy(); chips=m.chip_id.to_numpy()
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=a.seed).split(np.zeros(len(m)),yt,groups=chips))
def fitpred(X,y):  # train on tr, predict prob on te
    v=~np.all(X==0,1); tri=tr[v[tr]]
    sc=StandardScaler().fit(X[tri]); c=LogisticRegression(max_iter=4000,class_weight="balanced").fit(sc.transform(X[tri]),y[tri])
    p=np.full(len(te),np.nan); vt=v[te]; p[vt]=c.predict_proba(sc.transform(X[te][vt]))[:,1]; return p
def fitpred_rf(y):
    c=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=8,random_state=0).fit(spec[tr],y[tr])
    return c.predict_proba(spec[te])[:,1]
# predictions on test for each model x target
P={}
feats={f.split("/")[-1][len("features_"):-4]:f for f in sorted(glob.glob(f"{D}/features_*.npy"))+glob.glob(f"{ANY}/features_*.npy")}
for fm,f in feats.items():
    X=np.load(f)[ok]; P[(fm,"true")]=fitpred(X,yt); P[(fm,"proxy")]=fitpred(X,proxy)
P[("rf_spectral","true")]=fitpred_rf(yt); P[("rf_spectral","proxy")]=fitpred_rf(proxy)
te_chips=chips[te]; uchips=np.array(sorted(set(te_chips))); rng=np.random.default_rng(7)
yt_te=yt[te]; yp_te=proxy[te]
def metric(p,y,idx,kind):
    pp=p[idx]; yy=y[idx]; mask=~np.isnan(pp)
    pp,yy=pp[mask],yy[mask]
    if len(set(yy))<2: return np.nan
    return roc_auc_score(yy,pp) if kind=="auroc" else f1_score(yy,(pp>=.5).astype(int))
# bootstrap chip resamples -> index lists
chip_to_pos={c:np.where(te_chips==c)[0] for c in uchips}
boot_idx=[np.concatenate([chip_to_pos[c] for c in rng.choice(uchips,size=len(uchips),replace=True)]) for _ in range(a.B)]
def ci(vals): vals=[v for v in vals if not np.isnan(v)]; return (round(float(np.percentile(vals,2.5)),3),round(float(np.percentile(vals,97.5)),3))
res={"country":C,"B":a.B,"n_test_chips":int(len(uchips)),"models":{}}
print(f"[{C}] chip-clustered bootstrap B={a.B}, {len(uchips)} test chips")
for (fm,tg),p in P.items():
    y=yt_te if tg=="true" else yp_te
    for kind in ["auroc","f1"]:
        point=metric(p,y,np.arange(len(te)),kind)
        lo,hi=ci([metric(p,y,bi,kind) for bi in boot_idx])
        res["models"].setdefault(fm,{}).setdefault(tg,{})[kind]={"point":round(float(point),3),"ci":[lo,hi]}
# paired diffs (same resamples): RF proxy-F1 minus RF true-F1; bestFM true-AUROC minus RF true-AUROC
def paired(pa,ya,pb,yb,kind):
    d=[metric(pa,ya,bi,kind)-metric(pb,yb,bi,kind) for bi in boot_idx]
    return ci(d), round(float(metric(pa,ya,np.arange(len(te)),kind)-metric(pb,yb,np.arange(len(te)),kind)),3)
rf_t,rf_p=P[("rf_spectral","true")],P[("rf_spectral","proxy")]
(c1,p1)=paired(rf_p,yp_te,rf_t,yt_te,"f1"); res["paired_RFproxy_minus_RFtrue_f1"]={"point":p1,"ci":list(c1)}
# best FM on true by AUROC
fmnames=[k for k in feats]; bestfm=max(fmnames,key=lambda k:metric(P[(k,"true")],yt_te,np.arange(len(te)),"auroc"))
(c2,p2)=paired(P[(bestfm,"true")],yt_te,rf_t,yt_te,"auroc"); res["paired_bestFMtrue_minus_RFtrue_auroc"]={"best_fm":bestfm,"point":p2,"ci":list(c2)}
json.dump(res,open(f"data/results/ftw_bootstrap_ci_{C}.json","w"),indent=2)
for fm in res["models"]:
    t=res["models"][fm]["true"]; pr=res["models"][fm]["proxy"]
    print("  %-20s TRUE auroc %.3f%s f1 %.3f%s | PROXY auroc %.3f%s f1 %.3f%s"%(fm,
      t["auroc"]["point"],t["auroc"]["ci"],t["f1"]["point"],t["f1"]["ci"],
      pr["auroc"]["point"],pr["auroc"]["ci"],pr["f1"]["point"],pr["f1"]["ci"]))
print("  PAIRED RF(proxy-true) F1 diff = %.3f %s"%(p1,c1))
print("  PAIRED bestFM(%s) minus RF, TRUE AUROC = %.3f %s"%(bestfm,p2,c2))
print("wrote data/results/ftw_bootstrap_ci_%s.json"%C)
