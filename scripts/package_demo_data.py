"""Package the demo data bundle for the source-available release.

The tracked repo plus this bundle is enough to run the demo UI with no GPU
and no local model (cache-only mode): unzip into the checkout so the files
land under data/ and basis/, then start the server.

Includes exactly what ui/server.py reads at serve time for the default
(qwen, kernel) configuration. Never includes API keys, quarantine
directories, or stale backups; the zip is scanned after writing and the
script fails if anything forbidden slipped in.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "ammonix_rcm_demo_data.zip"

INCLUDE_DIRS = [
    "data/text_cache",
    "data/m1_ornith_ft",
    # the payer's reviewer readings: replaying the L-seat measurement offline
    "data/payer_reader",
    # the GPT-6 lane's cached decisions and writer payloads: the public demo
    # runs AMMONIX_LLM_LANE=gpt6 and must replay (and re-meter) keylessly
    "data/gpt6_ui_decisions",
    "data/m1_gpt6",
    "data/working",
    "basis/artefacts",
]
INCLUDE_FILES = [
    "data/universe_embed3d.parquet",
    "data/ui_rollout_kernel.json",
    "data/ui_curve_kernel.json",
    "data/ui_llm_lane.json",
    "data/ui_llm_lane_gpt6.json",
    "data/ui_llm_paperwork_gpt6.json",
]
FORBIDDEN = ("key", "quarantine", "stale", "secret", "token")


def forbidden(rel: str) -> bool:
    return any(marker in rel.lower() for marker in FORBIDDEN)


def main() -> int:
    entries: list[Path] = []
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            print(f"missing directory: {d}", file=sys.stderr)
            return 1
        entries.extend(p for p in sorted(base.rglob("*")) if p.is_file())
    for f in INCLUDE_FILES:
        p = ROOT / f
        if not p.is_file():
            print(f"missing file: {f}", file=sys.stderr)
            return 1
        entries.append(p)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in entries:
            rel = p.relative_to(ROOT).as_posix()
            if forbidden(rel):
                print(f"refusing to package: {rel}", file=sys.stderr)
                return 1
            zf.write(p, rel)

    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        bad = [n for n in names if forbidden(n)]
        if bad:
            OUT.unlink()
            print(f"forbidden entries found, bundle deleted: {bad}", file=sys.stderr)
            return 1

    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}: {len(names)} files, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
