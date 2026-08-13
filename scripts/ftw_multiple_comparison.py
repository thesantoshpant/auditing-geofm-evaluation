"""Multiple-comparison correction (Benjamini-Hochberg + Holm) across the 10
paired deltas (5 regions x {RFproxy-true F1, bestFM-RF true AUROC}). p-values
from a normal approximation to each chip-clustered bootstrap 95% CI
(se = (hi-lo)/(2*1.96)); two-sided."""
import json, math
from statistics import NormalDist
nd=NormalDist()
tests=[]
for C in ["india","kenya","vietnam","france","netherlands"]:
    d=json.load(open(f"data/results/ftw_bootstrap_ci_{C}.json"))
    for key,lbl in [("paired_RFproxy_minus_RFtrue_f1","RFproxy-true F1"),
                    ("paired_bestFMtrue_minus_RFtrue_auroc","bestFM-RF AUROC")]:
        v=d[key]; lo,hi=v["ci"]; pt=v["point"]; se=(hi-lo)/(2*1.959964)
        z=pt/se if se>0 else float("inf"); p=2*(1-nd.cdf(abs(z)))
        tests.append((f"{C}:{lbl}",pt,lo,hi,z,max(p,1e-300)))
tests.sort(key=lambda t:t[5])
m=len(tests)
# BH
bh_sig=[False]*m; thr=0
for i,(name,pt,lo,hi,z,p) in enumerate(tests,1):
    if p<=i/m*0.05: thr=i
for i in range(thr): bh_sig[i]=True
# Holm
holm_sig=[False]*m; 
for i,(name,pt,lo,hi,z,p) in enumerate(tests):
    holm_sig[i]= p<=0.05/(m-i)
print("10 paired deltas, sorted by p (normal-approx from bootstrap CI):")
out=[]
for i,(name,pt,lo,hi,z,p) in enumerate(tests):
    print("  %-28s delta=%+.3f [%.3f,%.3f] z=%.1f p=%.1e BH=%s Holm=%s"%(name,pt,lo,hi,z,p,bh_sig[i],holm_sig[i]))
    out.append({"test":name,"delta":pt,"ci":[lo,hi],"z":round(z,2),"p":p,"BH_sig_FDR0.05":bh_sig[i],"Holm_sig_0.05":holm_sig[i]})
allsig=all(bh_sig) and all(holm_sig)
print("\nALL 10 survive BH(FDR 0.05) AND Holm(0.05): %s ; min z = %.1f"%(allsig,min(t[4] for t in tests)))
with open(
    "data/results/ftw_multiple_comparison.json",
    "w",
    encoding="utf-8",
    newline="\n",
) as handle:
    json.dump(
        {
            "n_tests": m,
            "all_significant_BH_and_Holm": allsig,
            "min_z": round(min(t[4] for t in tests), 2),
            "tests": out,
        },
        handle,
        indent=2,
    )
print("wrote data/results/ftw_multiple_comparison.json")
