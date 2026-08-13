"""Edge control (proper): FM vs spectral RF on TRUE field-membership, evaluated
ONLY on boundary-zone pixels (both classes, sampled within 3px of a field
boundary). If FMs still beat RF here, the FM advantage is boundary-related
signal, not within-field interpolation of same-label neighbours.
"""
import glob, json, numpy as np, pandas as pd, rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

D="data/features_per_pixel_ftw_india_true_edge"
ANY="data/features_per_pixel_ftw_india_true_edge_anysat"
m=pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
s2f={f.split("/")[-1].rsplit("_s2.tif",1)[0]:f for f in glob.glob("data/chips_ftw_india/*/*_s2.tif")}
nb=None; spec=None
for cid,grp in m.groupby("chip_id"):
    if cid not in s2f: continue
    with rasterio.open(s2f[cid]) as s: arr=s.read()
    if nb is None: nb=arr.shape[0]; spec=np.full((len(m),nb),np.nan,dtype=np.float32)
    for i,r in grp.iterrows():
        rr,cc=int(r.pixel_r),int(r.pixel_c)
        if 0<=rr<arr.shape[1] and 0<=cc<arr.shape[2]: spec[i]=arr[:,rr,cc]
ok=(~np.isnan(spec).any(axis=1))&(~np.all(spec==0,axis=1))
ok=ok.to_numpy() if hasattr(ok,"to_numpy") else ok
m=m[ok].reset_index(drop=True); spec=spec[ok]; y=m.label.to_numpy(); chips=m.chip_id.to_numpy()
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=20260514).split(np.zeros(len(m)),y,groups=chips))
ov=len(set(chips[tr])&set(chips[te]))
print("EDGE-ZONE eval: %d boundary-zone px, pos_rate=%.3f, %d/%d chips, overlap=%d"%(len(m),y.mean(),len(set(chips[tr])),len(set(chips[te])),ov))
def lin(X):
    v=~np.all(X==0,axis=1); tri,tei=tr[v[tr]],te[v[te]]
    if len(set(y[tei]))<2: return None,None
    sc=StandardScaler().fit(X[tri]); c=LogisticRegression(max_iter=4000,class_weight="balanced").fit(sc.transform(X[tri]),y[tri])
    p=c.predict_proba(sc.transform(X[tei]))[:,1]; return round(roc_auc_score(y[tei],p),3),round(f1_score(y[tei],(p>=.5).astype(int)),3)
def rf():
    if len(set(y[te]))<2: return None,None
    c=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=8,random_state=0).fit(spec[tr],y[tr])
    p=c.predict_proba(spec[te])[:,1]; return round(roc_auc_score(y[te],p),3),round(f1_score(y[te],(p>=.5).astype(int)),3)
res={"n_boundary_px":int(len(m)),"pos_rate":round(float(y.mean()),3),"chip_overlap":int(ov),"models":{}}
print("%-22s | TRUE (boundary-zone) AUROC / F1"%"model"); print("-"*55)
for f in sorted(glob.glob(f"{D}/features_*.npy"))+glob.glob(f"{ANY}/features_*.npy"):
    fm=f.split("/")[-1][len("features_"):-4]; a,f1=lin(np.load(f)[ok]); res["models"][fm]={"auroc":a,"f1":f1}
    print("%-22s | %s / %s"%(fm,a,f1))
a,f1=rf(); res["models"]["rf_spectral"]={"auroc":a,"f1":f1}; print("%-22s | %s / %s"%("rf_spectral",a,f1))
json.dump(res,open("data/results/ftw_edge_zone_eval_india.json","w"),indent=2)
print("\nwrote data/results/ftw_edge_zone_eval_india.json")
