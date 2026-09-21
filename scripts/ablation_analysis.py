"""Full agent vs gates-off on the same draw: per-episode diff.

Writes runs/reports/ablation_paperwork_analysis.json with the counts the
paper and fig6 use."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "reports"

full = json.loads((R / "episodes_ammonix_comparison_paperwork_t99.json").read_text(encoding="utf-8"))
abl = json.loads((R / "episodes_ammonix_ablated_ablation_paperwork_t99.json").read_text(encoding="utf-8"))
assert set(full) == set(abl), "episode sets differ"

div = [e for e in full if full[e]["actions"] != abl[e]["actions"]]
helped = [e for e in div if full[e]["success"] and not abl[e]["success"]]
hurt = [e for e in div if abl[e]["success"] and not full[e]["success"]]


def tot(d, k):
    return round(sum(min(v["collected_payer"], v["allowed"]) if k == "capped"
                     else v[k] for v in d.values()), 2)


out = {
    "draw": "t99",
    "full": {
        "success": round(sum(v["success"] for v in full.values()) / len(full), 4),
        "collected_capped": tot(full, "capped"),
        "mistake_episodes": sum(1 for v in full.values() if v["mistakes"] > 0),
        "handed_off_touches": sum(v["handed_off"] for v in full.values()),
        "paperwork_rejected": sum(v["paperwork_rejected"] for v in full.values()),
    },
    "ablated": {
        "success": round(sum(v["success"] for v in abl.values()) / len(abl), 4),
        "collected_capped": tot(abl, "capped"),
        "mistake_episodes": sum(1 for v in abl.values() if v["mistakes"] > 0),
        "handed_off_touches": sum(v["handed_off"] for v in abl.values()),
        "paperwork_rejected": sum(v["paperwork_rejected"] for v in abl.values()),
    },
    "divergent_episodes": len(div),
    "gates_helped": len(helped),
    "gates_hurt": len(hurt),
    "helped_episodes": helped,
    "hurt_episodes": hurt,
}
path = R / "ablation_paperwork_analysis.json"
path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print(json.dumps({k: v for k, v in out.items() if not k.endswith("episodes")}, indent=1))
print("written:", path)
