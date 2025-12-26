from pathlib import Path

from vectortd.core.model.map import load_map_json
from vectortd.core.model.state import GameState
from vectortd.core.rules.wave_spawner import start_next_wave
from vectortd.core.rules.creep_motion import step_creeps


def test_creeps_progress_and_eventually_loop_decreasing_lives():
    m = load_map_json(Path("data/maps/dev_2lanes.json"))
    s = GameState(lives=3, score=0)

    start_next_wave(s, m)

    # Sanity: after a few ticks, at least one creep should have moved.
    x0, y0 = s.creeps[0].x, s.creeps[0].y
    for _ in range(10):
        step_creeps(s, m, dt_scale=1.0)
    assert (s.creeps[0].x, s.creeps[0].y) != (x0, y0)

    # Now run enough ticks to force at least one end-of-path loop.
    # Path is ~400px; speed ~1px/tick => a few hundred ticks is enough; give margin.
    for _ in range(5000):
        step_creeps(s, m, dt_scale=1.0)
        if s.lives < 3:
            break

    assert s.lives < 3, "At least one creep should reach end-of-path and decrement lives."
    assert all(c.path_point >= 0 for c in s.creeps)
