import json, os, glob
import torch
CKDIR="artifacts/metric_scale/checkpoints"; MFDIR="artifacts/metric_scale/manifests"; MTDIR="artifacts/metric_scale/metrics"

def load_variant(name):
    """Return (args, metrics) for a variant, preferring manifest, else ckpt, else metrics jsonl."""
    mf=f"{MFDIR}/{name}_manifest.json"
    if os.path.exists(mf):
        d=json.load(open(mf)); return d.get("args",{}), d.get("best_metrics",{})
    for ck in (f"{CKDIR}/{name}_best.pt", f"{CKDIR}/{name}.pt"):
        if os.path.exists(ck):
            c=torch.load(ck,map_location="cpu",weights_only=False); return c.get("args",{}), c.get("metrics",{}) or {}
    return {}, {}

def best_from_jsonl(name):
    """Min-median eval from the metrics jsonl (trajectory)."""
    p=f"{MTDIR}/{name}_eval.jsonl"
    if not os.path.exists(p): return None
    rows=[json.loads(l) for l in open(p) if l.strip()]
    rows=[r for r in rows if "median_abs_pct" in r]
    if not rows: return None
    return min(rows, key=lambda r:r["median_abs_pct"])

def depth(a): return "MoGe-2" if a.get("moge2_pointmap_dir") else "MoGe-v1"
def trained(a):
    if a.get("unfreeze_ss_backbone"): return "MoT backbone (pose fast + geom slow)"
    if a.get("unfreeze_slat_cross_attn"): return "head+dec + SLAT cross-attn (live)"
    if a.get("load_feature_cache") or a.get("cache_latents"): return "head+dec (cached, gen frozen)"
    return "head+dec (live, gen frozen)"
def loss(a):
    if a.get("unfreeze_ss_backbone"): return f"L_whd + trans(w={a.get('trans_loss_weight')}) + flow-L2SP(w={a.get('flow_loss_weight')})"
    dec=a.get("decoder","baseline")
    if dec in ("anchor","factored","binned"): return f"{dec}: scale(w={a.get('scale_loss_weight')})+prop(w={a.get('prop_loss_weight')})"
    if a.get("ss_ratio_loss_weight",0): return f"ss-ratio(w={a.get('ss_ratio_loss_weight')})"
    return "log-dim smooth_L1"
def lr(a):
    if a.get("unfreeze_ss_backbone"): return f"geom2e-6/pose1e-4"
    if a.get("unfreeze_slat_cross_attn"): return f"head {a.get('lr')}/slat {a.get('slat_lr')}"
    return str(a.get("lr"))
def data(a):
    s=a.get("omninocs_sources",[]); s="+".join(x.replace("_real275","").replace("scenes","")[:4] for x in s) if isinstance(s,list) else str(s)
    n=a.get("max_records_per_source") or a.get("max_records"); return f"{s} ({n}/src{' bal' if a.get('balanced_sampling') else ''})"
def src(m):
    s=m.get("source_mean_abs_pct",{}) or {}
    return (s.get("nocs_real275"), s.get("objectron"), s.get("arkitscenes"))

VARIANTS=["mixed_v1","mixed_omninocs_1024dim_bs512","mixed_omninocs_balanced_100_per_source",
 "mixed_omninocs_balanced_6358_per_source_bs512","mixed_scratch_10k",
 "nocs_sceneholdout_1024dim_baseline_v2","nocs_sceneholdout_slat_conditioned_v1",
 "nocs_sceneholdout_slat_conditioned_v3","nocs_sceneholdout_ss_ratio_v1",
 "mixed_moge2_v1","mixed_moge2_v2_long","mixed_moge2_anchor_v1","mixed_moge2_factored_v1",
 "mixed_moge2_binned_v2","nocs_moge2_v1","nocs_moge2_v2_long","mixed_moge2_live_v1","joint_mot_v1"]

print(f"| variant | depth | data | trained | LR | loss | epochs | mean% | median% | NOCS | Obj | ARKit |")
print(f"|---|---|---|---|---|---|---|---|---|---|---|---|")
for v in VARIANTS:
    a,m=load_variant(v)
    if not m or "median_abs_pct" not in m:
        j=best_from_jsonl(v)
        if j: m=j
    if not m: print(f"| {v} | ? | (no data) |||||||||"); continue
    n,o,ar=src(m)
    f=lambda x: f"{x:.1f}" if isinstance(x,(int,float)) else "—"
    print(f"| {v} | {depth(a)} | {data(a)} | {trained(a)} | {lr(a)} | {loss(a)} | {a.get('epochs')} | "
          f"{f(m.get('mean_abs_pct'))} | {f(m.get('median_abs_pct'))} | {f(n)} | {f(o)} | {f(ar)} |")
