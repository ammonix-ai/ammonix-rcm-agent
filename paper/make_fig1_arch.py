"""Decision-time architecture schematic (fig1): how one live claim flows
through the RCM Agent's components. Colors mirror the components table:
grey = hand-written rules, green = learned from data, blue = stored
records, peach = frozen language models, pink = the world."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
matplotlib.rcParams["text.parse_math"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures"); os.makedirs(FIG, exist_ok=True)

DATA = "#dceaf5"   # stored records
MODEL = "#d7efe6"  # learned from data
RULE = "#eeeeee"   # hand-written rules
FROZ = "#fdeeda"   # frozen language models
WORLD = "#f7e3ee"  # the world
INK = "#1a1a1a"; MUT = "#5f6368"; BLUE = "#0072b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "savefig.dpi": 200})

fig, ax = plt.subplots(figsize=(12.2, 7.9)); ax.set_xlim(0, 24); ax.set_ylim(0, 15.6); ax.axis("off")

def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                 fc=fc, ec=MUT, lw=1.1, zorder=2))
    ax.text(x+w/2, y+h-0.30, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK, zorder=3)
    ax.text(x+w/2, y+h-0.86, sub, ha="center", va="top", fontsize=7.3, color=MUT, zorder=3)

def arrow(x1, y1, x2, y2, c=INK, lw=1.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 connectionstyle=f"arc3,rad={rad}", lw=lw, color=c, ls=ls, zorder=1,
                 shrinkA=2, shrinkB=2))

# ── Row 1: the decision path (left -> right) ──
y1, h1, w1, gap = 11.4, 2.3, 4.2, 0.55
x1 = [0.4 + i*(w1+gap) for i in range(5)]
box(x1[0], y1, w1, h1, "Live claim", "state and payer\ncorrespondence", DATA)
box(x1[1], y1, w1, h1, "Mechanistic world model φ", "hand-written rules\n83 clear-text features", RULE)
box(x1[2], y1, w1, h1, "Classifier swarm Λ", "one classifier per action;\ncalibrated scores = coord. u", MODEL)
box(x1[3], y1, w1, h1, "Knowledge Universe + Ether", "nearest settled claims\nscore each action", DATA)
box(x1[4], y1, w1, h1, "Skills", "20 versioned rules:\nact, ask, or hand off", MODEL)
for i in range(4):
    arrow(x1[i]+w1, y1+h1/2, x1[i+1], y1+h1/2)

# ask / hand-off exit (to a person)
arrow(x1[4]+w1+0.05, y1+h1*0.62, x1[4]+w1+0.55, y1+h1*0.62, c=MUT, lw=1.2)
ax.text(x1[4]+w1+0.28, y1+h1*0.62+0.55, "to a\nperson", ha="center", va="bottom",
        fontsize=7.2, color=MUT, style="italic")

# ── Row 2: what feeds the decision (one column left, freeing the lane under Skills) ──
y2, h2, w2 = 7.6, 1.9, 4.2
box(x1[0], y2, w2, h2, "Hard-rule masks", "remove actions\nthat are not allowed", RULE)
box(x1[1], y2, w2, h2, "Uncertainty detector", "fold copies disagree\n= close call, ask first", MODEL)
box(x1[2], y2, w2, h2, "Expected-value close-out", "dead ends ranked\nin expected dollars", RULE)
box(x1[3], y2, w2, h2, "Case-based fallback", "similar past situations\ndecide low-confidence claims", DATA)
arrow(x1[0]+w2*0.7, y2+h2, x1[4]+w1*0.12, y1, rad=-0.10)
arrow(x1[1]+w2*0.7, y2+h2, x1[4]+w1*0.32, y1, rad=-0.08)
arrow(x1[2]+w2*0.7, y2+h2, x1[4]+w1*0.52, y1, rad=-0.06)
arrow(x1[3]+w2*0.7, y2+h2, x1[4]+w1*0.72, y1, rad=-0.04)

# ── Row 3: execution (right -> left) ──
y3, h3, w3 = 3.1, 2.3, 5.4
xL, xS, xP = 17.9, 11.4, 4.9
box(xL, y3, w3, h3, "L: frozen language models", "Ornith 9B writes the forms;\nQwen 27B answers the operator;\nneither decides", FROZ)
box(xS, y3, w3, h3, "Scrubber", "deterministic checks against\nthe case file; three drafts,\nthen a person", RULE)
box(xP, y3, w3, h3, "Payer (the world)", "adjudicates claims and\npaperwork; its reviewer\nreads the letters", WORLD)
arrow(x1[4]+w1*0.5, y1, xL+w3*0.5, y3+h3)                    # Skills -> L (decision)
ax.text((x1[4]+w1*0.5+xL+w3*0.5)/2-0.35, (y1+y3+h3)/2, "decision", ha="right", va="center",
        fontsize=7.8, color=MUT, style="italic")
arrow(xL, y3+h3/2, xS+w3, y3+h3/2)                           # L -> scrubber
arrow(xS, y3+h3/2, xP+w3, y3+h3/2)                           # scrubber -> payer

# ── Outcome returns to the store: L-shaped route up the free lane at x=4.87 ──
lane = x1[0]+w1+0.18    # between column 0 and column 1, clear of box pads
ytop = 14.55
ax.plot([lane, lane], [y3+h3-0.35, ytop], color=BLUE, lw=1.8, ls=(0, (6, 3)), zorder=1)
ax.plot([lane, x1[3]+w1*0.5], [ytop, ytop], color=BLUE, lw=1.8, ls=(0, (6, 3)), zorder=1)
arrow(x1[3]+w1*0.5, ytop, x1[3]+w1*0.5, y1+h1+0.05, c=BLUE, lw=1.8)
ax.text((lane+x1[3]+w1*0.5)/2, ytop+0.42,
        "outcome recorded: the settled claim joins the store; nothing is retrained",
        ha="center", va="center", fontsize=8.0, color=BLUE, fontweight="bold")

# ── Legend ──
yl = 0.55
items = [(RULE, "hand-written rules"), (MODEL, "learned from data"),
         (DATA, "stored records"), (FROZ, "frozen language models"), (WORLD, "the world")]
xoff = 2.2
for fc, lab in items:
    ax.add_patch(Rectangle((xoff, yl), 0.55, 0.55, fc=fc, ec=MUT, lw=0.9))
    ax.text(xoff+0.75, yl+0.28, lab, ha="left", va="center", fontsize=7.8, color=INK)
    xoff += 0.75 + len(lab)*0.145 + 1.0

out = os.path.join(FIG, "fig1_architecture.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
