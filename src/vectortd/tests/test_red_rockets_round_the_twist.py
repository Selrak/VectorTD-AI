from pathlib import Path

from vectortd.core.model.map import load_map_json
from vectortd.core.model.state import GameState
from vectortd.core.rules.creep_motion import step_creeps
from vectortd.core.rules.tower_attack import step_towers
from vectortd.testing.definitions import (
    TestRunner,
    apply_test_definition,
    load_test_definition,
    resolve_map_path,
)


def test_red_rockets_round_the_twist_definition() -> None:
    root = Path(__file__).resolve().parents[3]
    definition_path = root / "data/tests/red_rockets_round_the_twist.json"
    test_definition = load_test_definition(definition_path)
    map_path = resolve_map_path(test_definition.map, root)
    map_data = load_map_json(map_path)
    state = GameState()
    apply_test_definition(state, map_data, test_definition)

    runner = TestRunner(test_definition, state)
    while runner.status == "running":
        step_creeps(state, map_data, dt_scale=1.0)
        step_towers(state, map_data, dt_scale=1.0)
        runner.advance(1.0)

    assert runner.status == "passed"
    if test_definition.goal.min_lives is not None:
        assert state.lives >= test_definition.goal.min_lives
    else:
        assert state.lives == test_definition.starting_lives
