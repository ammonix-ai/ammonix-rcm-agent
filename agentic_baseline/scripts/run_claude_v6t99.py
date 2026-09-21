"""Run the Claude Sonnet 5 post-hoc arm on tranche 99, loading the API key
inside Python (utf-8-sig, stripped) so no shell quoting can corrupt it.
Thin wrapper over run_comparison.py --arms claude."""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FACTORY = BASE.parent  # co-located: baseline lives inside the factory repo
key = (FACTORY / "data" / "anthropic_key.txt").read_text(encoding="utf-8-sig").strip()
assert key.startswith("sk-ant-") and len(key) > 60, "key file malformed"
os.environ["ANTHROPIC_API_KEY"] = key
os.environ.setdefault("AMMONIX_CLAUDE_MODEL", "claude-opus-5")
os.environ.setdefault("AMMONIX_FACTORY_ROOT", str(FACTORY))
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.argv = ["run_comparison.py", "--arms", "claude", "--tranche", "99",
            "--report", "comparison_opus_v6t99.json", "--tag", "_opus_v6t99"]
sys.path.insert(0, str(BASE / "scripts"))
import runpy
runpy.run_path(str(BASE / "scripts" / "run_comparison.py"), run_name="__main__")
