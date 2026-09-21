"""Repo-root conftest: make `cardessa` and `ammonix_core` importable in tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
