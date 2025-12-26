from pathlib import Path

import pytest

from vectortd.core.model.map import load_map_json
from vectortd.core.model.state import GameState
from vectortd.core.rules.creep_motion import step_creeps
from vectortd.core.rules.wave_spawner import start_next_wave


def test_wave_spawn_count_is_paths_times_14():
    m = load_map_json(Path("data/maps/dev_2lanes.json"))
    s = GameState()
    start_next_wave(s, m)
    assert len(s.creeps) == len(m.paths) * 14


def test_wave_interest_applies_to_bank_and_score():
    m = load_map_json(Path("data/maps/dev_2lanes.json"))
    s = GameState(bank=199, interest=1, score=10)
    start_bank = s.bank
    start_score = s.score

    start_next_wave(s, m)

    bank_gain = int(start_bank / 100 * s.interest)
    expected_bank = start_bank + bank_gain
    score_gain = int(expected_bank / 100 * s.interest)
    expected_score = start_score + score_gain

    assert s.bank == expected_bank
    assert s.score == expected_score


def _run_creeps_for_seconds(state: GameState, map_data, seconds: float, fps: int = 60) -> None:
    ticks = int(seconds * fps)
    for _ in range(ticks):
        step_creeps(state, map_data, dt_scale=1.0)
        if state.game_over:
            break


@pytest.mark.parametrize(
    ("map_path", "expected_loss"),
    [
        ("data/maps/dev_line5.json", 1),
        ("data/maps/dev_2lanes.json", 2),
    ],
)
def test_lives_drop_after_10_seconds(map_path: str, expected_loss: int) -> None:
    m = load_map_json(Path(map_path))
    s = GameState()
    start_lives = s.lives

    start_next_wave(s, m)
    _run_creeps_for_seconds(s, m, seconds=10, fps=60)

    assert s.lives <= start_lives - expected_loss
