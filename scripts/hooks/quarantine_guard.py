"""PreToolUse hook: deny every tool access to data/quarantine/**.

Constrain, don't only review (platform plan v0.3 Section 2): the quarantine is
enforced at the tool boundary for all agents and subagents, not by convention.

Reads the hook payload JSON from stdin. Exit code 2 blocks the tool call and
feeds stderr back to the agent; exit code 0 allows it.

The only sanctioned path into quarantine is the final evaluation gate (L10 /
P10), run through an exact whitelisted command. Everything else is denied,
including probe reads, globs, greps and shell commands that reference the path.
"""

import json
import re
import sys

QUARANTINE_PATTERN = re.compile(r"data[\\/]+quarantine", re.IGNORECASE)

# Exact commands (after whitespace normalisation) allowed to touch quarantine.
# The final-eval script itself may read quarantine A exactly once; nothing else may.
WHITELISTED_COMMANDS = frozenset(
    {
        "make final-eval",
        "python scripts/final_eval.py",
    }
)

PATH_FIELDS = (
    "file_path",
    "path",
    "notebook_path",
    "pattern",
    "old_string",  # cheap defence against edits that rewrite quarantine paths
    "new_string",
)


def normalise(command: str) -> str:
    return " ".join(command.split())


def references_quarantine(tool_name: str, tool_input: dict) -> str | None:
    """Return the offending field name, or None when the call is clean."""
    if tool_name in ("Bash", "PowerShell"):
        command = str(tool_input.get("command", ""))
        if QUARANTINE_PATTERN.search(command):
            if normalise(command) in WHITELISTED_COMMANDS:
                return None
            return "command"
        return None
    for field in PATH_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and QUARANTINE_PATTERN.search(value):
            return field
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # malformed payload: fail open for non-tool traffic, hooks re-fire per call
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    offender = references_quarantine(tool_name, tool_input)
    if offender is None:
        return 0
    print(
        f"BLOCKED by quarantine guard: {tool_name}.{offender} references "
        "data/quarantine/. The quarantine test set is sealed; only the "
        "whitelisted final-eval path (L10/P10) may touch it, exactly once.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
