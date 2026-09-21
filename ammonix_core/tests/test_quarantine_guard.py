"""Negative tests for the quarantine PreToolUse hook.

The guard is rewarded for refusing: blocked probes are the passing behaviour
(platform plan v0.3, Section 5 point 4).
"""

import json
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "quarantine_guard.py"

ALLOW, BLOCK = 0, 2


def run_guard(tool_name: str, tool_input: dict) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    return subprocess.run(
        [sys.executable, str(GUARD)],
        input=payload,
        capture_output=True,
        text=True,
    )


def test_read_of_quarantine_is_blocked():
    result = run_guard("Read", {"file_path": "data/quarantine/episodes.parquet"})
    assert result.returncode == BLOCK
    assert "quarantine" in result.stderr.lower()


def test_windows_path_separator_is_blocked():
    result = run_guard("Read", {"file_path": "data\\quarantine\\episodes.parquet"})
    assert result.returncode == BLOCK


def test_absolute_path_is_blocked():
    result = run_guard(
        "Write",
        {"file_path": "C:/repo/data/quarantine/x.parquet", "content": "x"},
    )
    assert result.returncode == BLOCK


def test_glob_and_grep_probes_are_blocked():
    assert run_guard("Glob", {"pattern": "data/quarantine/**"}).returncode == BLOCK
    assert run_guard("Grep", {"pattern": "payer", "path": "data/quarantine"}).returncode == BLOCK


def test_shell_command_referencing_quarantine_is_blocked():
    for tool in ("Bash", "PowerShell"):
        result = run_guard(tool, {"command": "ls data/quarantine/"})
        assert result.returncode == BLOCK, tool


def test_whitelisted_final_eval_commands_are_allowed():
    assert run_guard("Bash", {"command": "make final-eval"}).returncode == ALLOW
    assert run_guard("Bash", {"command": "python scripts/final_eval.py"}).returncode == ALLOW


def test_disguised_final_eval_command_is_still_blocked():
    result = run_guard(
        "Bash", {"command": "python scripts/final_eval.py; cat data/quarantine/a.parquet"}
    )
    assert result.returncode == BLOCK


def test_ordinary_paths_are_allowed():
    assert run_guard("Read", {"file_path": "data/raw/cardessa_sim/states.parquet"}).returncode \
        == ALLOW
    assert run_guard("Bash", {"command": "pytest -q"}).returncode == ALLOW


def test_edit_rewriting_quarantine_path_is_blocked():
    result = run_guard(
        "Edit",
        {
            "file_path": "scripts/split.py",
            "old_string": "data/quarantine/",
            "new_string": "data/working/",
        },
    )
    assert result.returncode == BLOCK


def test_malformed_payload_fails_open():
    result = subprocess.run(
        [sys.executable, str(GUARD)], input="not json", capture_output=True, text=True
    )
    assert result.returncode == ALLOW
