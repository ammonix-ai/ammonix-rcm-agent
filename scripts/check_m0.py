"""M0 checker: pytest and ruff must both exit 0 in ammonix_core.

Runs both commands, shows both outputs, and ends by printing the one-line JSON
verdict the goal condition references (transcript-evidence rule, platform plan
v0.3 Section 2).
"""

import json
import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "ammonix_core"


def run(label: str, args: list[str]) -> int:
    print(f"$ {' '.join(args)}  (cwd={CORE})")
    result = subprocess.run(args, cwd=CORE)
    print(f"[{label} exit code: {result.returncode}]")
    return result.returncode


def main() -> int:
    pytest_rc = run("pytest", [sys.executable, "-m", "pytest", "-q"])
    ruff_rc = run("ruff", [sys.executable, "-m", "ruff", "check", "."])
    verdict = pytest_rc == 0 and ruff_rc == 0
    print(
        json.dumps(
            {"loop": "L0", "verdict": verdict, "pytest_exit": pytest_rc, "ruff_exit": ruff_rc}
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
