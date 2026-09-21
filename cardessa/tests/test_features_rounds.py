"""Proposal blocks for rounds 3+: derivations correct, composable, deterministic."""

import pytest

from cardessa.features_rounds import BLOCK_NAMES, build_with_blocks
from cardessa.tests.test_features import tiny_states


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown proposal blocks"):
        build_with_blocks(tiny_states(), {"nonsense"})


def test_derived_block_values():
    frame, specs = build_with_blocks(tiny_states(), {"derived"})
    by_id = {r["state_id"]: r for r in frame.to_dicts()}
    first = by_id["e1-0"]  # retro_window 10, days_since_service 3, auth required
    assert first["retro_window_remaining"] == 7.0
    assert first["retro_still_possible"] == 1.0
    assert first["filing_window_burned"] == pytest.approx(3 / 90)
    later = by_id["e1-1"]  # days_since_service 24 > retro window 10
    assert later["retro_window_remaining"] == -14.0
    assert later["retro_still_possible"] == 0.0
    other = by_id["e2-0"]  # no auth required -> retro not applicable
    assert other["retro_still_possible"] == 0.0
    assert other["balance_to_allowed"] == pytest.approx(120.0 / 95.0)
    names = [s.name for s in specs]
    assert "last_action=submit_clean" not in names  # other block not included


def test_last_event_block_values():
    frame, specs = build_with_blocks(tiny_states(), {"last_event"})
    by_id = {r["state_id"]: r for r in frame.to_dicts()}
    assert by_id["e1-0"]["is_first_touch"] == 1.0
    assert by_id["e1-1"]["is_first_touch"] == 0.0
    assert by_id["e1-1"]["last_action=submit_clean"] == 1.0
    assert by_id["e1-1"]["last_carc=CO-197"] == 1.0
    assert by_id["e2-0"]["last_carc=CO-197"] == 0.0


def test_blocks_compose_and_stay_deterministic():
    states = tiny_states()
    frame, specs = build_with_blocks(states, set(BLOCK_NAMES))
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    assert "retro_still_possible" in names and "is_first_touch" in names
    rebuilt, _ = build_with_blocks(states, set(BLOCK_NAMES))
    assert frame.equals(rebuilt)
    feature_frame = frame.select(names)
    assert feature_frame.null_count().sum_horizontal().item() == 0
