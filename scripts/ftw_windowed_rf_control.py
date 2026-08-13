"""Spatial-baseline control: a windowed spectral RF (5x5 neighbourhood of the 12 S2
bands = 300-d) on the SAME controlled pixels/split as the per-pixel RF and the FMs,
TRUE field-membership label. Tests whether the FM advantage is merely 'spatial pooling':
if this spatially-pooled baseline still loses badly to FMs, it is not."""
import sys; sys.path.insert(0,"scripts")
import json, glob, numpy as np, pandas as pd, rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
out={}
for C in ["india","kenya","vietnam","france","netherlands"]:
    D=f"data/features_per_pixel_ftw_{C}_true"
    m=pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
    s2f={f.split("/")[-1].rsplit("_s2.tif",1)[0]:f for f in glob.glob(f"data/chips_ftw_{C}/*/*_s2.tif")}
    K=2  # 5x5
    feat=np.full((len(m),12*(2*K+1)**2),np.nan,np.float32)
    for cid,grp in m.groupby("chip_id"):
        if cid not in s2f: continue
        with rasterio.open(s2f[cid]) as s: arr=s.read().astype(np.float32)  # 12,H,W
        pad=np.pad(arr,((0,0),(K,K),(K,K)),mode="reflect")
        for i,r in grp.iterrows():
            rr,cc=int(r.pixel_r)+K,int(r.pixel_c)+K
            feat[i]=pad[:,rr-K:rr+K+1,cc-K:cc+K+1].reshape(-1)
    y=m.label.to_numpy(); chips=m.chip_id.to_numpy()
    ok=(~np.isnan(feat).any(1))&(~np.all(feat==0,1))
    feat,y2,chips2=feat[ok],y[ok],chips[ok]
    tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=20260514).split(np.zeros(len(y2)),y2,groups=chips2))
    c=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=8,random_state=0).fit(feat[tr],y2[tr])
    p=c.predict_proba(feat[te])[:,1]
    win_auc=round(roc_auc_score(y2[te],p),3); win_f1=round(f1_score(y2[te],(p>=.5).astype(int)),3)
    j=json.load(open(f"data/results/ftw_controlled_label_comparison{'' if C=='india' else '_'+C}.json"))["models"]
    perpix=j["rf_spectral"]["true"]["auroc"]
    bestfm=max((j[k]["true"]["auroc"] for k in j if k!="rf_spectral"))
    out[C]={"perpix_rf_true_auroc":perpix,"windowed_rf_true_auroc":win_auc,"windowed_rf_true_f1":win_f1,"best_fm_true_auroc":round(bestfm,3),"n_test":int(len(te))}
    print("%-12s perpix-RF %.3f  WINDOWED-RF %.3f  best-FM %.3f"%(C,perpix,win_auc,bestfm))
json.dump(out,open("data/results/ftw_windowed_rf_control.json","w"),indent=2)
print("wrote data/results/ftw_windowed_rf_control.json")
