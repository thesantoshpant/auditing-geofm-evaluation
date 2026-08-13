"""Edge-pixel control for the within-chip spatial-autocorrelation objection.
On the TRUE field-membership task, restrict the TEST set to pixels within k px
of a field/non-field boundary (where 'copy my neighbours' interpolation is
ambiguous). Train on the full train split as usual; evaluate only on near-edge
test pixels. If the FM advantage over the spectral RF PERSISTS on edge pixels,
the FM win reflects boundary-related structure, not interior label smoothing.
Reuses cached TRUE-mode features (no GPU).
"""
import sys, glob, json, numpy as np, pandas as pd, rasterio
sys.path.insert(0, "scripts")
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from extract_features_per_pixel import build_polygon_id_raster
import geopandas as gpd
from collections import defaultdict

D="data/features_per_pixel_ftw_india_true"
m=pd.read_parquet(f"{D}/features_per_pixel_meta.parquet").reset_index(drop=True)
polys=gpd.read_parquet("data/labels/polygons_ftw_india_keyed.parquet")
pby=defaultdict(list)
for _,p in polys.iterrows():
    pby[p["chip_id"]].append({"polygon_id":p["polygon_id"],"district":p["district"],"geometry":p["geometry"],"area_m2":float(p["area_m2"]),"size_bin":p["size_bin"]})
s2f={f.split("/")[-1].rsplit("_s2.tif",1)[0]:f for f in glob.glob("data/chips_ftw_india/*/*_s2.tif")}

# distance-to-boundary + raw S2 per sampled pixel
dist=np.full(len(m),-1.0); spec=None; nb=None
for cid,grp in m.groupby("chip_id"):
    if cid not in s2f: continue
    pid_raster,_=build_polygon_id_raster(s2f[cid], pby.get(cid,[]))
    fld=(pid_raster>0)
    # boundary = field pixels adjacent to non-field (and vice versa)
    er=ndimage.binary_erosion(fld); dil=ndimage.binary_dilation(fld)
    boundary=(fld & ~er) | (~fld & dil)
    dt=ndimage.distance_transform_edt(~boundary)  # px distance to nearest boundary
    with rasterio.open(s2f[cid]) as s: arr=s.read()
    if nb is None: nb=arr.shape[0]; spec=np.full((len(m),nb),np.nan,dtype=np.float32)
    for i,r in grp.iterrows():
        rr,cc=int(r.pixel_r),int(r.pixel_c)
        if 0<=rr<dt.shape[0] and 0<=cc<dt.shape[1]:
            dist[i]=dt[rr,cc]; spec[i]=arr[:,rr,cc]
ok=(dist>=0)&(~np.isnan(spec).any(axis=1))&(~np.all(spec==0,axis=1))
m=m[ok].reset_index(drop=True); spec=spec[ok]; dist=dist[ok]
y=m.label.to_numpy(); chips=m.chip_id.to_numpy()
tr,te=next(GroupShuffleSplit(1,test_size=0.25,random_state=20260514).split(np.zeros(len(m)),y,groups=chips))
print("edge ablation: %d px; median dist-to-boundary=%.1f px"%(len(m),np.median(dist)))
def ev_lin(X,te_sub):
    v=~np.all(X==0,axis=1); tri=tr[v[tr]]; tei=te_sub[v[te_sub]]
    if len(set(y[tei]))<2: return None,None
    sc=StandardScaler().fit(X[tri]); c=LogisticRegression(max_iter=4000,class_weight="balanced").fit(sc.transform(X[tri]),y[tri])
    p=c.predict_proba(sc.transform(X[tei]))[:,1]; return round(roc_auc_score(y[tei],p),3),round(f1_score(y[tei],(p>=.5).astype(int)),3)
def ev_rf(te_sub):
    if len(set(y[te_sub]))<2: return None,None
    c=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=8,random_state=0).fit(spec[tr],y[tr])
    p=c.predict_proba(spec[te_sub])[:,1]; return round(roc_auc_score(y[te_sub],p),3),round(f1_score(y[te_sub],(p>=.5).astype(int)),3)
res={}
for tag,thr in [("ALL test",1e9),("near-edge (<=2px)",2.0),("near-edge (<=1px)",1.0)]:
    te_sub=te[dist[te]<=thr]
    n1=int(y[te_sub].sum()); n0=int((y[te_sub]==0).sum())
    print("\n=== %s : n_test=%d (pos=%d neg=%d) ==="%(tag,len(te_sub),n1,n0))
    row={}
    for f in sorted(glob.glob(f"{D}/features_*.npy")):
        fm=f.split("/")[-1][len("features_"):-4]; a,f1=ev_lin(np.load(f)[ok],te_sub); row[fm]={"auroc":a,"f1":f1}; print("  %-22s AUROC=%s F1=%s"%(fm,a,f1))
    a,f1=ev_rf(te_sub); row["rf_spectral"]={"auroc":a,"f1":f1}; print("  %-22s AUROC=%s F1=%s"%("rf_spectral",a,f1))
    res[tag]={"n_test":len(te_sub),"pos":n1,"neg":n0,"models":row}
json.dump(res,open("data/results/ftw_edge_pixel_ablation.json","w"),indent=2)
print("\nwrote data/results/ftw_edge_pixel_ablation.json")
