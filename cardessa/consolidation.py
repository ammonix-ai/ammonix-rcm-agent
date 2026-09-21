"""v0.6 M2: consolidation policy for the kernel decision field.

The kernel field (cardessa.kernel_field) can absorb a newly adjudicated
record without retraining any model. Left ungoverned, that makes the deployed
system's behaviour drift record by record, which is exactly what the
factory's determinism posture forbids. This module is the governance:

- consolidate(): the one sanctioned way a record enters the field. It only
  accepts a record whose episode is terminal and whose success label has
  been adjudicated by the outcome function (never a live case: temporal
  firewall), stamps it with the adjudication ordinal, and appends it.
- Recency window: MAX_RECORDS keeps the field bounded; when exceeded, the
  oldest consolidated records (by adjudication ordinal) age out first. The
  build-time universe records carry ordinal 0 and are never aged out by
  default (they are the pinned basis); set AGE_OUT_BASIS to allow it.
- Snapshots: snapshot() writes the field's arrays plus a manifest entry
  {version, n_records, n_consolidated, content_hash, parent_hash, stamp}
  to basis/kernel_snapshots/. A deployment pins ONE snapshot hash; the
  live field may grow between snapshots, but any decision can be attributed
  to "snapshot H plus N consolidated records", both hash-recorded.
- load_snapshot(): restores a field bit-identically from a snapshot.

None of this activates the field. Activation is the runtime switch in
harness.BasisRuntime.load(...) / AMMONIX_DECISION_FIELD, gated by
runs/APPROVALS.md.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cardessa.kernel_field import KernelField

MAX_RECORDS = 50_000
AGE_OUT_BASIS = False


class ConsolidationLog:
    """Ordinal stamps per record; ordinal 0 = build-time basis."""

    def __init__(self, n_basis: int):
        self.ordinals = np.zeros(n_basis, dtype=np.int64)
        self.next_ordinal = 1

    def stamp(self) -> int:
        o = self.next_ordinal
        self.next_ordinal += 1
        self.ordinals = np.append(self.ordinals, o)
        return o


def consolidate(field: KernelField, log: ConsolidationLog, *, u_row,
                action: str, success: bool, state_id: str,
                terminal: bool, adjudicated: bool) -> int:
    """Append one adjudicated record; returns its ordinal. Refuses live cases."""
    if not (terminal and adjudicated):
        raise ValueError(
            f"{state_id}: only terminal, adjudicated records may be consolidated "
            "(temporal firewall)"
        )
    field.insert(u_row, action, success, state_id)
    ordinal = log.stamp()
    _enforce_window(field, log)
    return ordinal


def _enforce_window(field: KernelField, log: ConsolidationLog) -> None:
    n = len(field.state_ids)
    if n <= MAX_RECORDS:
        return
    excess = n - MAX_RECORDS
    order = np.argsort(log.ordinals, kind="stable")  # oldest first
    if not AGE_OUT_BASIS:
        order = order[log.ordinals[order] > 0]
    drop = set(order[:excess].tolist())
    keep = np.asarray([i for i in range(n) if i not in drop])
    field.U = field.U[keep]
    field.state_ids = [field.state_ids[i] for i in keep]
    field.actions = field.actions[keep]
    field.success = field.success[keep]
    log.ordinals = log.ordinals[keep]
    field._refit()


def snapshot(field: KernelField, log: ConsolidationLog, root: Path,
             version: str, parent_hash: str | None = None) -> dict:
    out = Path(root) / "basis" / "kernel_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    h = field.content_hash()
    stem = out / f"kernel_{version}_{h[:12]}"
    np.save(f"{stem}.U.npy", field.U)
    np.save(f"{stem}.success.npy", field.success)
    np.save(f"{stem}.ordinals.npy", log.ordinals)
    (Path(f"{stem}.meta.json")).write_text(json.dumps({
        "state_ids": field.state_ids, "actions": field.actions.tolist(),
        "u_order": field.u_order, "next_ordinal": int(log.next_ordinal),
    }), encoding="utf-8", newline="\n")
    entry = {
        "version": version, "content_hash": h, "parent_hash": parent_hash,
        "n_records": len(field.state_ids),
        "n_consolidated": int((log.ordinals > 0).sum()),
        "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [f"{stem.name}.U.npy", f"{stem.name}.success.npy",
                  f"{stem.name}.ordinals.npy", f"{stem.name}.meta.json"],
    }
    manifest = out / "manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
    entries.append(entry)
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8", newline="\n")
    return entry


def load_snapshot(root: Path, content_hash: str):
    out = Path(root) / "basis" / "kernel_snapshots"
    entries = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in entries if e["content_hash"] == content_hash)
    stem = out / entry["files"][0].replace(".U.npy", "")
    meta = json.loads(Path(f"{stem}.meta.json").read_text(encoding="utf-8"))
    field = KernelField(
        U=np.load(f"{stem}.U.npy"), state_ids=meta["state_ids"],
        actions=meta["actions"], success=np.load(f"{stem}.success.npy"),
        u_order=meta["u_order"],
    )
    log = ConsolidationLog(0)
    log.ordinals = np.load(f"{stem}.ordinals.npy")
    log.next_ordinal = meta["next_ordinal"]
    assert field.content_hash() == content_hash, "snapshot hash mismatch"
    return field, log
