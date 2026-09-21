"""Regenerate fig_c2_case.png alone, with the state named as the deployed
interface names it (ep-05035-s1)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
detail = os.path.join(ROOT, "runs", "reports", "c2_case_detail.json")
d = json.load(open(detail, encoding="utf-8"))
if d["state_id"] == "ep-05035-r1":
    d["state_id"] = "ep-05035-s1"  # the same demo state under the shipped naming
    json.dump(d, open(detail, "w", encoding="utf-8"), indent=1)
    print("state id updated")

src = open(os.path.join(ROOT, "paper", "make_figures.py"), encoding="utf-8").read()
lines = src.split("\n")
header = "\n".join(lines[:40])
i = next(k for k, l in enumerate(lines) if "c2_case_detail.json" in l) - 1
j = next(k for k, l in enumerate(lines) if "fig_c2_case ok" in l)
block = "\n".join(lines[i:j + 1])
exec(compile(header + "\n" + block, "figc2", "exec"))
