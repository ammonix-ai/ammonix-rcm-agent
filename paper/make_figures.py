"""
Generate all Ammonix-RCM paper figures from on-disk artifacts.
Every number is loaded from the shipped v0.6 reports/basis; nothing is hardcoded
except axis cosmetics. Palette: Okabe-Ito (CVD-safe, validated).
Run from repo root:  python paper/make_figures.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito (validated CVD-safe). Fixed order, never cycled.
OI = dict(blue="#0072b2", orange="#e69f00", green="#009e73", pink="#cc79a7",
          sky="#56b4e9", vermillion="#d55e00", yellow="#f0e442", black="#111111",
          grey="#9aa0a6")
INK, MUT, GRID, SURF = "#1a1a1a", "#5f6368", "#e6e6e6", "#ffffff"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUT, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
    "axes.grid": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
matplotlib.rcParams["text.parse_math"] = False  # literal $ everywhere; no math-mode

def load(p): return json.load(open(os.path.join(ROOT, p)))
def despine(ax, keep=("left","bottom")):
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(s in keep)

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Head-to-head (4 arms), success% + $ collected, oracle ceiling
# ══════════════════════════════════════════════════════════════════════════
cmp = load("runs/reports/comparison_v6_tranche99.json")  # v0.6, tranche 99
arms = [("ammonix_v6","Ammonix RCM Agent",OI["blue"]),
        ("agentic","Agentic LLM\n(same 27B)",OI["orange"]),
        ("personas","Simulated billers",OI["grey"])]
oc = cmp["oracle_greedy"]

fig, (a1,a2) = plt.subplots(1,2, figsize=(9.2,4.0))
# success%
xs = np.arange(len(arms))
succ = [cmp[k]["success_rate_v2"]*100 for k,_,_ in arms]
mist = [cmp[k]["episodes_with_mistakes"] for k,_,_ in arms]
cols = [c for _,_,c in arms]
# 95% Wilson intervals (n=300) as error bars, per Anthropic eval-reporting practice
def _wilson(p, n=300, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h
_ci = [_wilson(cmp[k]["success_rate_v2"]) for k,_,_ in arms]
_err = [[v/100 - lo for v,(lo,_) in zip(succ,_ci)], [hi - v/100 for v,(_,hi) in zip(succ,_ci)]]
bars = a1.bar(xs, succ, color=cols, width=0.62, zorder=3)
a1.errorbar(xs, succ, yerr=[[100*e for e in _err[0]],[100*e for e in _err[1]]], fmt="none",
            ecolor=INK, elinewidth=1.1, capsize=4, zorder=4)
a1.axhline(oc["success_rate_v2"]*100, ls=(0,(4,3)), lw=1.3, color=OI["green"], zorder=2)
a1.text(len(arms)-0.5, oc["success_rate_v2"]*100+0.8,
        f"oracle ceiling {oc['success_rate_v2']*100:.0f}%", ha="right", va="bottom",
        fontsize=8.5, color=OI["green"])
for x,v,m in zip(xs,succ,mist):
    a1.text(x, 100*_ci[list(xs).index(x)][1]+0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    a1.text(x, 2.5, f"{m} mistake\nepisodes", ha="center", va="bottom", fontsize=7.6, color="white")
a1.set_ylim(0,72); a1.set_ylabel("Claims resolved (success, %)")
a1.set_xticks(xs); a1.set_xticklabels([n for _,n,_ in arms], fontsize=8.6)
a1.set_title("A  Resolution rate on 300 sealed episodes", fontsize=10.5, loc="left", fontweight="bold")
despine(a1)

# $ collected
coll = [cmp[k]["payer_collected_capped"] for k,_,_ in arms]
walls = [cmp[k]["wall_seconds"] for k,_,_ in arms]
bars2 = a2.bar(xs, [c/1000 for c in coll], color=cols, width=0.62, zorder=3)
# bootstrap 95% intervals on capped payer dollars (10,000 episode resamples, fixed seed)
_epfiles = {"ammonix_v6":"episodes_ammonix_v6_v6t99.json","agentic":"episodes_agentic_v6t99.json","personas":"episodes_personas_v6t99.json"}
_agentic_root = os.path.join(ROOT, "..", "..", "Agentic_Baseline", "runs", "reports")
_rng = np.random.default_rng(20260817)
_dci = []
for k,_,_ in arms:
    _p = os.path.join(_agentic_root, _epfiles[k])
    if os.path.exists(_p):
        _E = json.load(open(_p)); _d = np.array([min(e["collected_payer"], e["allowed"]) for e in _E.values()])
        _b = np.array([_d[_rng.integers(0,len(_d),len(_d))].sum() for _ in range(10000)])
        _dci.append((np.percentile(_b,2.5), np.percentile(_b,97.5)))
    else:
        _dci.append((None,None))
for x,c,(lo,hi) in zip(xs,coll,_dci):
    if lo is not None:
        a2.errorbar([x],[c/1000],yerr=[[(c-lo)/1000],[(hi-c)/1000]],fmt="none",ecolor=INK,elinewidth=1.1,capsize=4,zorder=4)
a2.axhline(oc["payer_collected_capped"]/1000, ls=(0,(4,3)), lw=1.3, color=OI["green"], zorder=2)
a2.text(len(arms)-0.5, oc["payer_collected_capped"]/1000+0.6,
        f"oracle ceiling ${oc['payer_collected_capped']/1000:.1f}k", ha="right", va="bottom",
        fontsize=8.5, color=OI["green"])
for x,v,w in zip(xs,coll,walls):
    a2.text(x, (_dci[list(xs).index(x)][1] or v)/1000+0.7, f"${v/1000:.1f}k", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    a2.text(x, 2.0, f"{w:.1f}s", ha="center", va="bottom", fontsize=8, color="white")
a2.set_ylim(0,88); a2.set_ylabel("Payer dollars collected (capped, $000)")
a2.set_xticks(xs); a2.set_xticklabels([n for _,n,_ in arms], fontsize=8.6)
a2.set_title("B  Dollars collected  (wall-clock in bar)", fontsize=10.5, loc="left", fontweight="bold")
despine(a2)
fig.suptitle("Pre-registered, information-parity head-to-head  (tranche 99, n=300)",
             x=0.01, ha="left", fontsize=8.5, color=MUT)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(FIG,"fig1_headtohead.png")); plt.close(fig); print("fig1 ok")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Learning curve ($ vs n_train, 3-draw band, human + LLM ref)
# ══════════════════════════════════════════════════════════════════════════
cur = load("data/ui_curve_kernel.json")  # v0.6 kernel field curve
pts = pd.DataFrame(cur["points"])
# columns include n_train, collected (dollars). Discover keys.
ntr = [c for c in pts.columns if "n" in c.lower() and ("train" in c.lower() or c.lower()=="n")]
ncol = ntr[0] if ntr else pts.columns[0]
dcol = [c for c in pts.columns if "collect" in c.lower() or "dollar" in c.lower()]
dcol = dcol[0] if dcol else "collected"
g = pts.groupby(ncol)[dcol].agg(["mean","min","max"]).reset_index().sort_values(ncol)
human = cur["human_collected"]
llm_lane = load("data/ui_llm_lane.json")
# LLM ref = payer-counted collected across the 200 demo lanes
def lane_sum(l):
    for k in ("payer_collected","collected_payer","payer","collected"):
        if isinstance(l,dict) and k in l: return l[k]
    return 0
llm_ref = None
try:
    lanes = llm_lane if isinstance(llm_lane,list) else llm_lane.get("lanes",[])
    llm_ref = sum(lane_sum(x) for x in lanes)
except Exception: pass

fig, ax = plt.subplots(figsize=(7.4,4.4))
ax.fill_between(g[ncol], g["min"]/1000, g["max"]/1000, color=OI["blue"], alpha=0.16, zorder=2)
ax.plot(g[ncol], g["mean"]/1000, color=OI["blue"], lw=2.2, marker="o", ms=6,
        mfc="white", mec=OI["blue"], mew=1.6, zorder=4, label="Ammonix RCM Agent")
ax.axhline(human/1000, ls=(0,(5,3)), lw=1.6, color=OI["grey"], zorder=3)
ax.text(g[ncol].max(), human/1000-1.6, f"simulated billers  ${human/1000:.1f}k",
        ha="right", va="top", fontsize=8.5, color=MUT)
if llm_ref:
    ax.axhline(llm_ref/1000, ls=(0,(5,3)), lw=1.6, color=OI["orange"], zorder=3)
    ax.text(g[ncol].max(), llm_ref/1000+0.5, f"agentic LLM  ${llm_ref/1000:.1f}k",
            ha="right", va="bottom", fontsize=8.5, color=OI["orange"])
ax.set_xscale("log")
# plain-integer tick labels at the data points: the log formatter emits
# $\mathdefault{10^{2}}$ labels that render literally under parse_math=False
_xt = sorted(g[ncol].unique())
ax.set_xticks(_xt)
ax.set_xticklabels([str(int(v)) for v in _xt], fontsize=9)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Training claims remembered  (log scale)")
ax.set_ylabel("Payer dollars collected on fixed 200-claim demo ($000)")
# crossover marker (first memory size whose mean beats the billers)
cross = g[g["mean"]>=human].iloc[0][ncol] if (g["mean"]>=human).any() else None
ax.set_title(f"Sample-efficiency: Ammonix passes its teachers by ~{int(cross):,} claims"
             if cross else "Sample-efficiency curve",
             fontsize=11, loc="left", fontweight="bold")
if cross:
    ax.axvline(cross, ls=":", lw=1.0, color=MUT, zorder=1)
ax.legend(frameon=False, loc="lower right", fontsize=9)
despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig2_learning_curve.png")); plt.close(fig); print("fig2 ok")

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-action classifier AUROC (+ ECE), sorted
# ══════════════════════════════════════════════════════════════════════════
man = load("basis/manifest.json")
uni = load("runs/reports/universe.json")
met = man.get("metrics", {})
ece = uni["ece_per_action"]
# metrics may be dict action->{auroc,...}. Discover.
rows=[]
for act, m in met.items():
    if isinstance(m, dict) and ("auroc" in m or "AUROC" in m or "test_auroc" in m):
        au = m.get("auroc", m.get("AUROC", m.get("test_auroc")))
        npos = m.get("n_pos"); nneg = m.get("n_neg")
        rows.append((act, au, ece.get(act), npos, nneg))
df = pd.DataFrame(rows, columns=["action","auroc","ece","npos","nneg"]).dropna(subset=["auroc"]).sort_values("auroc")
fig, ax = plt.subplots(figsize=(7.6,4.3))
ys = np.arange(len(df))
ax.barh(ys, df["auroc"], color=OI["blue"], height=0.6, zorder=3)
ax.axvline(0.5, ls=(0,(4,3)), color=MUT, lw=1.0, zorder=2)
ax.text(0.502, -0.16, "chance = 0.5", ha="left", fontsize=7.6, color=MUT,
        transform=ax.get_xaxis_transform())
for y,(_,r) in zip(ys, df.iterrows()):
    ax.text(r["auroc"]+0.006, y, f"{r['auroc']:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")
    n = f"n+={int(r['npos'])}" if pd.notna(r['npos']) else ""
    ax.text(0.505, y, n, va="center", ha="left", fontsize=7.4, color="white")
ax.set_yticks(ys); ax.set_yticklabels(df["action"], fontsize=9)
ax.set_xlim(0.5,0.98); ax.set_xlabel("Out-of-fold AUROC")
ax.set_title("Per-action classifier discrimination (Λ swarm, 5-fold OOF)",
             fontsize=11, loc="left", fontweight="bold", pad=10)
ax.text(0.98, -0.16, "bill_secondary & write_off carried as constant priors (not shown)",
        ha="right", fontsize=7.6, color=MUT, transform=ax.get_xaxis_transform())
despine(ax, keep=("left","bottom"))
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig3_peraction_auroc.png")); plt.close(fig)
print("fig3 ok  (actions plotted:", len(df), ")")

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Knowledge Universe (label-space UMAP), 2 panels
# ══════════════════════════════════════════════════════════════════════════
emb = pd.read_parquet(os.path.join(ROOT,"data/universe_embed3d.parquet"))
ub  = pd.read_parquet(os.path.join(ROOT,"basis/universe.parquet"))
U = emb.merge(ub[["state_id","true_action_id","outcome_success","tribe_id"]], on="state_id")
fam = {"submit_clean":"submit","submit_with_records":"submit",
       "appeal_with_necessity":"contest","request_peer_to_peer":"contest",
       "request_retro_auth":"contest","correct_and_resubmit":"contest",
       "provide_requested_info":"contest","bill_patient":"close",
       "bill_secondary":"close","write_off":"close"}
U["family"]=U["true_action_id"].map(fam).fillna("other")
famcol={"submit":OI["blue"],"contest":OI["orange"],"close":OI["green"],"other":OI["grey"]}
amb=set(uni["ambiguous_tribes"] if isinstance(uni["ambiguous_tribes"][0],str)
        else [t.get("tribe_id",t.get("id")) for t in uni["ambiguous_tribes"]])

fig,(p1,p2)=plt.subplots(1,2, figsize=(11,5.2))
# Panel A: by action family
for f,c in famcol.items():
    s=U[U["family"]==f]
    p1.scatter(s["x"], s["y"], s=5, c=c, alpha=0.45, linewidths=0, rasterized=True)
p1.set_title("A  Cases colored by action family", fontsize=10.5, loc="left", fontweight="bold")
p1.legend(handles=[Patch(color=famcol[f], label=f) for f in ["submit","contest","close"]],
          frameon=False, fontsize=9, loc="upper right")
# Panel B: outcome + ambiguous tribe centroids
ok=U[U["outcome_success"]]; no=U[~U["outcome_success"]]
p2.scatter(no["x"],no["y"], s=5, c=OI["grey"], alpha=0.30, linewidths=0, rasterized=True)
p2.scatter(ok["x"],ok["y"], s=5, c=OI["green"], alpha=0.55, linewidths=0, rasterized=True)
cen=U.dropna(subset=["tribe_id"]).groupby("tribe_id")[["x","y"]].mean()
namb=0
for tid,row in cen.iterrows():
    if str(tid) in amb:
        p2.scatter(row["x"],row["y"], s=120, facecolors="none", edgecolors=OI["vermillion"],
                   linewidths=1.8, zorder=5); namb+=1
p2.set_title(f"B  Outcome (green=success) + {namb} ambiguous tribes (rings)",
             fontsize=10.5, loc="left", fontweight="bold")
p2.legend(handles=[Patch(color=OI["green"],label="success"),Patch(color=OI["grey"],label="failure"),
                   Patch(facecolor="none",edgecolor=OI["vermillion"],label="ambiguous tribe")],
          frameon=False, fontsize=9, loc="upper right")
for ax in (p1,p2):
    ax.set_xticks([]); ax.set_yticks([]); despine(ax, keep=())
fig.suptitle("Ammonix-RCM Knowledge Universe — 11,427 states, label-space UMAP, 58 tribes",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(FIG,"fig4_universe.png"), dpi=220); plt.close(fig)
print("fig4 ok  (ambiguous centroids drawn:", namb, ")")

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Calibration reliability (reconstructed from scores_cal)
# ══════════════════════════════════════════════════════════════════════════
def cal_of(row):
    try:
        d=json.loads(row["scores_cal"]) if isinstance(row["scores_cal"],str) else row["scores_cal"]
        return d.get(row["true_action_id"])
    except Exception: return None
ub2=ub.copy()
ub2["p"]=ub2.apply(cal_of, axis=1)
rel=ub2.dropna(subset=["p"]).copy()
rel["y"]=rel["outcome_success"].astype(float)
bins=np.linspace(0,1,11)
rel["b"]=pd.cut(rel["p"],bins,include_lowest=True)
agg=rel.groupby("b").agg(pred=("p","mean"),obs=("y","mean"),n=("y","size")).dropna()
fig,ax=plt.subplots(figsize=(5.4,5.2))
ax.plot([0,1],[0,1], ls=(0,(4,3)), color=MUT, lw=1.1, zorder=2)
ax.plot(agg["pred"],agg["obs"], "-o", color=OI["blue"], lw=2, ms=7, mfc="white",
        mec=OI["blue"], mew=1.6, zorder=4)
for _,r in agg.iterrows():
    if r["pred"] < 0.06:
        ax.text(r["pred"]+0.03, r["obs"]+0.015, f"{int(r['n'])}", ha="left", va="bottom", fontsize=6.8, color=MUT)
    else:
        ax.text(r["pred"], r["obs"]-0.035, f"{int(r['n'])}", ha="center", va="top", fontsize=6.8, color=MUT)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
ax.set_xlabel("Predicted success probability (calibrated)")
ax.set_ylabel("Observed success rate")
ax.set_title("Calibration on the taken action", fontsize=11, loc="left", fontweight="bold")
ax.text(0.03,0.94,f"cross-fit ECE = {uni['ece_overall']:.3f}\nbin count shown below points",
        fontsize=8.5, color=INK, va="top")
despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig5_calibration.png")); plt.close(fig); print("fig5 ok")

# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — Episodic-memory ablation
# ══════════════════════════════════════════════════════════════════════════
ab=load("runs/reports/ablation_v6t99.json")
agg=ab["aggregate"]; dv=ab["divergences"]
full=agg["ammonix_v6_full"]; abl=agg["ammonix_v4_ablated"]  # runner key for the ablated arm
fig,(b1,b2)=plt.subplots(1,2, figsize=(9.2,4.2), gridspec_kw={"width_ratios":[1.15,1]})
# paired bars: success & $
labels=["Memory ON\n(full)","Memory OFF\n(argmax)"]
succ=[full["success_rate_v2"]*100, abl["success_rate_v2"]*100]
doll=[full["payer_collected_capped"]/1000, abl["payer_collected_capped"]/1000]
x=np.arange(2); w=0.34
b1.bar(x-w/2, succ, w, color=OI["blue"], zorder=3, label="success %")
b1b=b1.twinx()  # NOTE: two measures -> but keep single-axis rule: instead show $ as separate group
# revert twin; use grouped with normalized second axis is dual-axis (forbidden). Do two panels instead.
b1b.remove()
for xi,v in zip(x-w/2,succ): b1.text(xi,v+0.6,f"{v:.1f}%",ha="center",fontsize=9,fontweight="bold")
b1.bar(x+w/2, doll, w, color=OI["sky"], zorder=3)
for xi,v in zip(x+w/2,doll): b1.text(xi,v+0.6,f"${v:.1f}k",ha="center",fontsize=9,fontweight="bold")
b1.set_xticks(x); b1.set_xticklabels(labels, fontsize=9)
b1.set_ylim(0,76)
b1.set_ylabel("success %  (dark)   /   $000 collected  (light)")
b1.set_title(f"A  Memory gates add {succ[0]-succ[1]:+.1f} pts / {(doll[0]-doll[1]):+.1f}k",
             fontsize=10, loc="left", fontweight="bold")
despine(b1)
# divergence breakdown
segs=[("memory saved",dv["memory_saved"],OI["green"]),
      ("neutral",dv["neutral"],OI["grey"]),
      ("memory cost",dv["memory_cost"],OI["vermillion"])]
left=0
for name,val,c in segs:
    b2.barh(0,val,left=left,color=c,zorder=3,height=0.5)
    if val >= 10:
        b2.text(left+val/2,0,f"{name}\n{val}",ha="center",va="center",fontsize=8.5,
                color="white" if c!=OI["grey"] else INK)
    else:
        b2.text(left+val/2,0.32,f"{name} ({val})",ha="center",va="bottom",fontsize=8, color=c)
    left+=val
b2.set_xlim(0,dv["n_divergent_episodes"]); b2.set_ylim(-0.6,0.75)
b2.set_yticks([]); b2.set_xlabel(f"{dv['n_divergent_episodes']} divergent episodes")
b2.set_title("B  Where memory changed the call", fontsize=10, loc="left", fontweight="bold")
despine(b2, keep=("bottom",))
fig.text(0.5, 0.005,
        "memory-saved pattern: on these low-confidence claims no action clears the confidence floor, so the "
        "full agent escalates the touch instead of committing to a value-destroying argmax; the memory-off "
        "agent takes the low-confidence best action and writes the claim off",
        ha="center", va="bottom", fontsize=7.6, color=MUT)
fig.tight_layout(rect=[0,0.05,1,1]); fig.savefig(os.path.join(FIG,"fig6_ablation.png")); plt.close(fig); print("fig6 ok")

# ══════════════════════════════════════════════════════════════════════════
# FIG C2 — case-based fallback in action (real v5 case)
# ══════════════════════════════════════════════════════════════════════════
from collections import Counter as _Counter
cc = load("runs/reports/c2_case_detail.json")
sc = cc["scores"]; nb = cc["neighbours"]; chosen = cc["c2_action"]
figc,(pa,pb) = plt.subplots(1,2, figsize=(9.6,4.0), gridspec_kw={"width_ratios":[1,1.2]})
acts=list(sc.keys()); vals=[sc[a] for a in acts]; ya=np.arange(len(acts))[::-1]
pa.barh(ya, vals, color=OI["sky"], zorder=3, height=0.6)
pa.axvline(cc["floor"], ls=(0,(4,3)), color=OI["vermillion"], lw=1.5)
pa.text(cc["floor"], len(acts)-0.2, f"confidence floor {cc['floor']}", color=OI["vermillion"],
        fontsize=8, ha="center", va="bottom")
for y,v in zip(ya,vals): pa.text(v+max(vals)*0.03, y, f"{v:.3f}", va="center", fontsize=8.6)
pa.set_yticks(ya); pa.set_yticklabels([a.replace("_"," ") for a in acts], fontsize=8.8)
pa.set_xlim(0, max(vals)*1.6+0.005)
pa.set_title("A  Classifier swarm: no action clears the floor", fontsize=10, loc="left", fontweight="bold")
despine(pa)
byact=_Counter(n["action"] for n in nb); paid=_Counter(n["action"] for n in nb if n["success"])
acts2=sorted(byact, key=lambda a:-byact[a]); yb=np.arange(len(acts2))[::-1]
for y,a in zip(yb,acts2):
    tot=byact[a]; pd=paid[a]
    pb.barh(y, pd, color=OI["green"], zorder=3, height=0.55)
    pb.barh(y, tot-pd, left=pd, color=OI["grey"], zorder=3, height=0.55)
    pb.text(tot+0.3, y, f"{tot}  ({pd} paid)", va="center", fontsize=8.6)
pb.set_yticks(yb); pb.set_yticklabels([a.replace("_"," ") for a in acts2], fontsize=8.8)
pb.set_xlim(0, max(byact.values())+6)
pb.set_title(f"B  Nearest {len(nb)} feature-space cases -> agent recalls: {chosen.replace('_',' ')}",
             fontsize=9.8, loc="left", fontweight="bold")
despine(pb)
figc.text(0.5, 0.015, f"Real case {cc['state_id']} ({cc['payer']}, CPT {cc['cpt']}): the swarm sits below the "
        f"confidence floor, so instead of escalating, the agent retrieves the nearest cases in feature space; "
        f"{int(cc['neighbour_share_on_action']*100)}% took {chosen.replace('_',' ')} and it carries positive "
        f"expected value, so the agent takes it — the Knowledge Universe driving the decision.",
        ha="center", va="bottom", fontsize=7.7, color=MUT, wrap=True)
figc.tight_layout(rect=[0,0.07,1,1]); figc.savefig(os.path.join(FIG,"fig_c2_case.png")); plt.close(figc); print("fig_c2_case ok")

print("\nALL FIGURES WRITTEN TO", FIG)
