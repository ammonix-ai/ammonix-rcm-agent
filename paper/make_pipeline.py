"""Data-build pipeline schematic for Ammonix-RCM (fig9). Real stage names/counts."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams["text.parse_math"]=False
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG=os.path.join(ROOT,"paper","figures"); os.makedirs(FIG,exist_ok=True)

OI=dict(blue="#0072b2",orange="#e69f00",green="#009e73",pink="#cc79a7",sky="#56b4e9",
        vermillion="#d55e00",grey="#9aa0a6")
DATA="#dceaf5"; MODEL="#d7efe6"; RULE="#eeeeee"; INK="#1a1a1a"; MUT="#5f6368"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":200,"savefig.dpi":200})

fig,ax=plt.subplots(figsize=(12.2,6.4)); ax.set_xlim(0,24); ax.set_ylim(0,13); ax.axis("off")

def box(x,y,w,h,title,sub,fc,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.18",
                 fc=fc,ec=MUT,lw=1.1,zorder=2))
    ax.text(x+w/2,y+h-0.34,title,ha="center",va="top",fontsize=9.4,fontweight="bold",color=tc,zorder=3)
    ax.text(x+w/2,y+h-0.86,sub,ha="center",va="top",fontsize=7.6,color=MUT,zorder=3)

def arrow(x1,y1,x2,y2,c=INK):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,
                 lw=1.5,color=c,zorder=1,shrinkA=2,shrinkB=2))

# ── Row A: data generation (y ~ 9.3) ──
ya=9.3; w=4.4; h=2.1; gap=0.9
xs=[0.4+i*(w+gap) for i in range(4)]
box(xs[0],ya,w,h,"1  World fixtures","12 payers (2 OOD)\n25 clinics · 1,000 patients",DATA)
box(xs[1],ya,w,h,"2  Payer policy engine","deterministic per-payer rules\nreviewer reads appeal letters",DATA)
box(xs[2],ya,w,h,"3  Personas drive episodes","Diligent .40 / Hasty .35 / Consv .25\n1–7 touches per claim",DATA)
box(xs[3],ya,w,h,"4  Outcome adjudication","success = ≥90% allowed / 120d\noutcome v2: payer-valid only",DATA)
for i in range(3): arrow(xs[i]+w,ya+h/2,xs[i+1],ya+h/2)

# ── Row B: model build (y ~ 5.4) ──
yb=5.4; wb=3.55; hb=2.1; gapb=0.62
xb=[0.4+i*(wb+gapb) for i in range(5)]
box(xb[0],yb,wb,hb,"5  Tokenizer φ","83 features, hash-pinned\nno post-hoc facts",MODEL)
box(xb[1],yb,wb,hb,"6  Split + quarantine","patient-grouped 5-fold\nquarantine A sealed by hook",MODEL)
box(xb[2],yb,wb,hb,"7  Λ swarm + calibration","per-action GBDT, OOF\nisotonic, cross-fit ECE 0.024",MODEL)
box(xb[3],yb,wb,hb,"8  Knowledge Universe","11,427 states, label-space\nHDBSCAN → 58 tribes (9 amb.)",MODEL)
box(xb[4],yb,wb,hb,"9  Skills S + frozen L","RHO-VR dual-gate → 20 skills\n9B writes · 27B answers",MODEL)
for i in range(4): arrow(xb[i]+wb,yb+hb/2,xb[i+1],yb+hb/2)
# connector Row A -> Row B
arrow(xs[3]+w/2,ya,xb[0]+wb/2,yb+hb)
ax.text(xb[0]+wb/2+0.2,(ya+yb+hb)/2-0.1,"ingest + leak screen\n(poisoned column caught)",
        ha="left",va="center",fontsize=7.2,color=MUT,style="italic")

# ── RHO-VR retrospective loop: one clear arc under the whole build row ──
ax.add_patch(FancyArrowPatch((xb[4]+wb/2,yb-0.05),(xb[0]+wb/2,yb-0.05),
             connectionstyle="arc3,rad=0.28",arrowstyle="-|>",mutation_scale=15,
             lw=1.8,color=OI["orange"],ls=(0,(6,3)),zorder=1))
ax.text((xb[0]+xb[4])/2+wb/2,yb-1.9,
        "RHO-VR loop: propose harness update → score on held-out folds → keep only on strict improvement",
        ha="center",va="center",fontsize=8.0,color=OI["orange"],fontweight="bold")

# ── Bottom bands ──
yc=1.35
ax.add_patch(FancyBboxPatch((0.4,yc),11.0,1.5,boxstyle="round,pad=0.02,rounding_size=0.15",
             fc="#fdf0e6",ec=OI["vermillion"],lw=1.0,zorder=2))
ax.text(0.7,yc+1.15,"Planted instrumentation",fontsize=8.6,fontweight="bold",color=OI["vermillion"],va="top")
ax.text(0.7,yc+0.72,"payer-conditional trap (retro-auth) · poisoned column days_to_payment\n"
        "20 impossible cases · 2 out-of-distribution payers",fontsize=7.4,color=MUT,va="top")
ax.add_patch(FancyBboxPatch((12.0,yc),11.6,1.5,boxstyle="round,pad=0.02,rounding_size=0.15",
             fc="#eef1f4",ec=MUT,lw=1.0,zorder=2))
ax.text(12.3,yc+1.15,"Provenance & control",fontsize=8.6,fontweight="bold",color=INK,va="top")
ax.text(12.3,yc+0.72,"master-seed regeneration · content-hash manifests · maker/checker split\n"
        "three human gates (action-map · skill activation · release)",fontsize=7.4,color=MUT,va="top")

ax.text(0.4,12.5,"How the Ammonix-RCM world and agent are built",fontsize=12.5,fontweight="bold",color=INK)
ax.text(0.4,12.0,"synthetic data generation (top) → harness build under RHO-VR (middle) → firewall & provenance (bottom)",
        fontsize=8.4,color=MUT)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig9_pipeline.png"),bbox_inches="tight"); print("fig9 ok")
