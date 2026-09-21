"""fig6: the safeguards switched off, under the payer-adjudicated world.

Reads runs/reports/ablation_paperwork_analysis.json, writes
paper/figures/fig6_ablation.png (same file name the paper includes)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
OI = dict(blue="#0072b2", sky="#56b4e9", green="#009e73", grey="#999999",
          vermillion="#d55e00")
INK, MUT = "#1a1a1a", "#5f6368"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUT, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "axes.axisbelow": True,
})


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


ab = json.load(open(os.path.join(ROOT, "runs/reports/ablation_paperwork_analysis.json")))
full, abl = ab["full"], ab["ablated"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
labels = ["Safeguards ON\n(full)", "Safeguards OFF\n(argmax)"]
succ = [full["success"] * 100, abl["success"] * 100]
doll = [full["collected_capped"] / 1000, abl["collected_capped"] / 1000]
x = np.arange(2)
w = 0.34
b1.bar(x - w / 2, succ, w, color=OI["blue"], zorder=3)
for xi, v in zip(x - w / 2, succ):
    b1.text(xi, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
b1.bar(x + w / 2, doll, w, color=OI["sky"], zorder=3)
for xi, v in zip(x + w / 2, doll):
    b1.text(xi, v + 0.6, f"${v:.1f}k", ha="center", fontsize=9, fontweight="bold")
b1.set_xticks(x)
b1.set_xticklabels(labels, fontsize=9)
b1.set_ylim(0, 76)
b1.set_ylabel("success %  (dark)   /   $000 collected  (light)")
b1.set_title(f"A  The safeguards add {succ[0]-succ[1]:+.1f} pts / {doll[0]-doll[1]:+.1f}k",
             fontsize=10, loc="left", fontweight="bold")
despine(b1)

helped, hurt = ab["gates_helped"], ab["gates_hurt"]
neutral = ab["divergent_episodes"] - helped - hurt
segs = [("safeguards saved", helped, OI["green"]), ("neutral", neutral, OI["grey"]),
        ("safeguards cost", hurt, OI["vermillion"])]
left = 0
for name, val, c in segs:
    b2.barh(0, val, left=left, color=c, zorder=3, height=0.5)
    if val >= 12:
        b2.text(left + val / 2, 0, f"{name}\n{val}", ha="center", va="center",
                fontsize=8.5, color="white" if c != OI["grey"] else INK)
    else:
        b2.text(left + val / 2, 0.32, f"{name} ({val})", ha="center", va="bottom",
                fontsize=8, color=c)
    left += val
b2.set_xlim(0, ab["divergent_episodes"])
b2.set_ylim(-0.6, 0.75)
b2.set_yticks([])
b2.set_xlabel(f"{ab['divergent_episodes']} divergent episodes")
b2.set_title("B  Where the safeguards changed the call", fontsize=10, loc="left",
             fontweight="bold")
despine(b2, keep=("bottom",))
fig.text(0.5, 0.005,
         "safeguards-saved pattern: on these claims no action clears the confidence floor or the scores are "
         "nearly tied, so the full agent hands the touch to a clerk instead of committing to a "
         "value-destroying best guess; the agent without safeguards takes it and the balance is lost",
         ha="center", va="bottom", fontsize=7.6, color=MUT)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(os.path.join(FIG, "fig6_ablation.png"))
print("fig6 ok:", os.path.join(FIG, "fig6_ablation.png"))
