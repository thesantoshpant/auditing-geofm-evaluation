"""Persistent artifact for the proxy-mismatch mechanism: per region, over the
manifest chips, compute (a) true-field coverage fraction, (b) WorldCover-cropland
(class40) fraction, (c) fraction of true-NEGATIVE (non-field) pixels that WC
calls cropland. Explains why the proxy inflates the spectral RF everywhere.
"""
import sys, json, glob, numpy as np, rasterio
sys.path.insert(0,"scripts")
from extract_features_per_pixel import build_polygon_id_raster
import geopandas as gpd
from collections import defaultdict
out={}
for C in ["india","kenya","vietnam","france","netherlands"]:
    polyf=f"data/labels/polygons_ftw_{C}_keyed.parquet"
    try: polys=gpd.read_parquet(polyf)
    except Exception: continue
    pby=defaultdict(list)
    for _,p in polys.iterrows(): pby[p["chip_id"]].append({"polygon_id":p["polygon_id"],"district":p.get("district",""),"geometry":p["geometry"],"area_m2":float(p["area_m2"]),"size_bin":p["size_bin"]})
    s2={f.split("/")[-1].rsplit("_s2.tif",1)[0]:f for f in glob.glob(f"data/chips_ftw_{C}/*/*_s2.tif")}
    wcf={"_".join(f.split("/")[-1].split("_")[:3]):f for f in glob.glob(f"data/chips_ftw_{C}/*/*_worldcover.tif")}
    nf=ncrop=nneg=nnegcrop=ntot=0
    chips=[c for c in pby if c in s2 and c in wcf]
    import random; random.seed(0)
    for cid in chips[:250]:
        pid,_=build_polygon_id_raster(s2[cid], pby[cid]); fld=(pid>0)
        with rasterio.open(wcf[cid]) as s: wc=s.read(1)
        if wc.shape!=fld.shape: continue
        crop=(wc==40)
        nf+=int(fld.sum()); ncrop+=int(crop.sum()); ntot+=fld.size
        neg=~fld; nneg+=int(neg.sum()); nnegcrop+=int((neg&crop).sum())
    out[C]={"n_chips":len(chips[:250]),"true_field_coverage":round(nf/ntot,3),
            "wc_cropland_frac":round(ncrop/ntot,3),
            "trueneg_that_are_wc_cropland":round(nnegcrop/nneg,3)}
    print("%-12s coverage=%.3f  WCcrop=%.3f  trueNEG-as-WCcrop=%.3f  (n=%d chips)"%(C,out[C]["true_field_coverage"],out[C]["wc_cropland_frac"],out[C]["trueneg_that_are_wc_cropland"],out[C]["n_chips"]))
json.dump(out,open("data/results/ftw_proxy_mismatch_coverage.json","w"),indent=2)
print("wrote data/results/ftw_proxy_mismatch_coverage.json")
