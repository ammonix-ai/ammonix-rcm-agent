"""The life of one claim, A to Z: agent side and payer side, every model
in its place, with the three loops (scrubber returns, payer rejections,
the settled claim joining the store)."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
matplotlib.rcParams["text.parse_math"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures"); os.makedirs(FIG, exist_ok=True)

DATA = "#dceaf5"; MODEL = "#d7efe6"; RULE = "#efefef"; FROZ = "#fdeeda"; WORLD = "#f7e3ee"
AGENTBG = "#f4f6fc"; PAYERBG = "#fdf4f9"
INK = "#1a1a1a"; MUT = "#5f6368"; BLUE = "#0072b2"; RED = "#c0392b"; GREEN = "#1e8e5a"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "savefig.dpi": 200})

fig, ax = plt.subplots(figsize=(12.6, 7.4)); ax.set_xlim(0, 25); ax.set_ylim(0, 14.2); ax.axis("off")

# ── territories ──
ax.add_patch(FancyBboxPatch((0.25, 0.5), 13.9, 12.6, boxstyle="round,pad=0.02,rounding_size=0.25",
             fc=AGENTBG, ec="#b9c4e0", lw=1.2, zorder=0))
ax.add_patch(FancyBboxPatch((14.85, 0.5), 9.9, 12.6, boxstyle="round,pad=0.02,rounding_size=0.25",
             fc=PAYERBG, ec="#e3b9d2", lw=1.2, zorder=0))
ax.text(7.2, 12.55, "AMMONIX  ·  one workstation", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#3b4d8f")
ax.text(19.8, 12.55, "THE PAYER  ·  the world", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#a44a7b")
ax.plot([14.5, 14.5], [0.7, 12.4], color=MUT, lw=1.2, ls=(0, (5, 4)), zorder=0)
ax.text(14.5, 0.35, "the institutional boundary", ha="center", va="top", fontsize=7.2,
        color=MUT, style="italic")

def station(x, y, w, h, num, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                 fc=fc, ec=MUT, lw=1.1, zorder=2))
    ax.add_patch(Circle((x+0.02, y+h-0.02), 0.38, fc=INK, ec="white", lw=1.4, zorder=4))
    ax.text(x+0.02, y+h-0.04, num, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=5)
    ax.text(x+w/2+0.15, y+h-0.30, title, ha="center", va="top", fontsize=8.8,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x+w/2, y+h-0.86, sub, ha="center", va="top", fontsize=7.2, color=MUT, zorder=3)

def arr(x1, y1, x2, y2, c=INK, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 connectionstyle=f"arc3,rad={rad}", lw=lw, color=c, ls=ls, zorder=1,
                 shrinkA=3, shrinkB=3))

# ── agent side ──
station(0.8, 9.4, 4.0, 2.3, "1", "A claim arrives", "a denial is on file;\nthe balance is open", DATA)
station(5.6, 9.4, 4.0, 2.3, "2", "Ammonix decides", "counts how similar past\nclaims ended; picks the action;\nno language model", MODEL)
station(10.2, 9.4, 3.9, 2.3, "3", "Ornith 9B writes", "the appeal letter or form,\nfrom the case record alone", FROZ)
station(10.2, 5.5, 3.9, 2.3, "4", "The scrubber checks", "code, not a model: form vs\ncase file; three drafts,\nthen a person", RULE)
station(0.8, 1.3, 4.0, 2.2, "i", "The operator asks why", "Qwen 27B answers from\nthe record; it never decides", FROZ)
station(5.6, 1.3, 4.0, 2.2, "8", "The store grows", "the settled claim is stored;\nnothing is retrained", DATA)

# ── payer side ──
station(15.3, 9.4, 4.4, 2.3, "5", "The mailroom checks", "codes vs the records it holds;\nthe right denial answered;\nonly what was asked for", RULE)
station(20.0, 9.4, 4.5, 2.3, "6", "Haiku reads the letter", "every statement checked against\nthe records; only its\nverdict is used", FROZ)
station(17.6, 5.5, 4.4, 2.3, "7", "The payer answers", "pays, denies, or asks;\nits letter is written\nby Qwen 27B", WORLD)

# ── the happy path ──
arr(4.8, 10.55, 5.6, 10.55)                       # 1 -> 2
arr(9.6, 10.55, 10.2, 10.55)                      # 2 -> 3
arr(12.15, 9.4, 12.15, 7.8)                       # 3 -> 4
arr(13.9, 7.8, 15.6, 9.4, rad=-0.2)               # 4 -> 5 (crosses boundary)
ax.text(14.0, 8.85, "submitted", ha="center", va="center", fontsize=6.8,
        color=MUT, style="italic", rotation=52)
arr(19.7, 10.55, 20.0, 10.55)                     # 5 -> 6 (letters to the reviewer)
arr(22.2, 9.4, 20.7, 7.8, rad=0.15)               # 6 -> 7
arr(17.2, 9.4, 18.7, 7.8, rad=-0.15, c=MUT, lw=1.2)  # 5 -> 7 (forms go straight)

# ── loop: scrubber returns a bad draft ──
arr(11.3, 7.8, 11.3, 9.4, c=RED, lw=1.3, rad=0.35, ls=(0, (4, 3)))
ax.text(9.6, 8.55, "returned with\nthe reason", ha="center", va="center", fontsize=6.8, color=RED)

# ── loop: payer rejects paperwork, hugging the boundary ──
ax.plot([16.2, 16.2], [9.35, 6.9], color=RED, lw=1.3, ls=(0, (4, 3)), zorder=1)
arr(16.2, 6.9, 14.18, 6.9, c=RED, lw=1.3, ls=(0, (4, 3)))
ax.text(16.45, 8.6, "back unprocessed:\na touch and days lost", ha="left", va="center",
        fontsize=6.8, color=RED)

# ── the answer comes home (blue lane) ──
ax.plot([17.6, 15.8], [6.15, 4.4], color=BLUE, lw=1.7, zorder=1)
ax.plot([15.8, 8.4], [4.4, 4.4], color=BLUE, lw=1.7, zorder=1)
arr(8.4, 4.4, 7.4, 9.35, c=BLUE, lw=1.7, rad=0.12)
ax.text(11.55, 3.95, "the answer comes home: the claim continues,\nback to step 2 for the next touch (1 to 7 in all)",
        ha="center", va="center", fontsize=7.4, color=BLUE)

# ── or the claim settles (green lane to the store) ──
ax.plot([18.7, 18.7], [5.45, 2.4], color=GREEN, lw=1.7, zorder=1)
arr(18.7, 2.4, 9.75, 2.4, c=GREEN, lw=1.7)
ax.text(14.2, 2.85, "or the claim settles: paid, patient-billed, or written off",
        ha="center", va="center", fontsize=7.4, color=GREEN)

# ── the store informs the next decision ──
arr(6.6, 3.55, 6.6, 9.35, c=GREEN, lw=1.7, ls=(0, (6, 3)))
ax.text(5.3, 6.4, "counts from the\nnext decision on", ha="right", va="center",
        fontsize=6.8, color=GREEN)

out = os.path.join(FIG, "fig_journey.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
