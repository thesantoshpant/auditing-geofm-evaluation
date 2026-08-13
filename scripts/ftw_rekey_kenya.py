import json, geopandas as gpd, pandas as pd
idx={}
for ln in open("data/index/ftw_kenya.jsonl"):
    r=json.loads(ln); idx[r["aoi_id"]]="%.5f_%.5f_%d"%(r["lon"],r["lat"],int(r["scene_date"])//1000)
man=set(json.loads(l)["chip_id"] for l in open("data/chips/manifest_ftw_kenya.jsonl"))
g=gpd.read_parquet("data/labels/polygons_ftw_kenya.parquet")
g["chip_id"]=g["chip_id"].map(idx)
before=len(g)
g=g[g["chip_id"].notna() & g["chip_id"].isin(man)].copy()
g["polygon_id"]=[f"{c}_{i}" for i,c in enumerate(g["chip_id"])]
g.to_parquet("data/labels/polygons_ftw_kenya_keyed.parquet")
print("kenya rekey: %d->%d polygons, %d chips overlap manifest (of %d manifest chips)"%(
    before,len(g),g["chip_id"].nunique(),len(man)))
print("size bins:", g["size_bin"].value_counts().to_dict())
