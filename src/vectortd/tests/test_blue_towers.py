from types import SimpleNamespace

from vectortd.core.model.entities import Creep
from vectortd.core.model.state import GameState
from vectortd.core.rules.placement import place_tower
from vectortd.core.rules.tower_attack import step_towers


def _make_map():
    return SimpleNamespace(width=600, height=500, grid=25, paths=[])


def _make_creep(
    *,
    x: float,
    y: float,
    speed: float,
    max_speed: float,
    hp: float = 10000.0,
    type_id: int = 1,
) -> Creep:
    return Creep(
        x=float(x),
        y=float(y),
        type_id=int(type_id),
        path=[0],
        path_point=0,
        targ_x=float(x),
        targ_y=float(y),
        xval=0.0,
        yval=0.0,
        worth=1,
        speed=float(speed),
        max_speed=float(max_speed),
        hp=float(hp),
        maxhp=float(hp),
    )


def test_blue1_targets_first_four_fast_creeps() -> None:
    state = GameState()
    map_data = _make_map()
    state.bank = 99999
    tower = place_tower(state, map_data, 0, 0, "blue1")
    assert tower is not None
    base_damage = float(tower.damage)

    creeps = [
        _make_creep(x=20, y=20, speed=1.0, max_speed=1.0),
        _make_creep(x=25, y=20, speed=0.5, max_speed=1.0),
        _make_creep(x=30, y=20, speed=1.0, max_speed=1.0),
        _make_creep(x=35, y=20, speed=1.0, max_speed=1.0),
        _make_creep(x=40, y=20, speed=1.0, max_speed=1.0),
        _make_creep(x=45, y=20, speed=1.0, max_speed=1.0),
    ]
    state.creeps.extend(creeps)

    step_towers(state, map_data, dt_scale=1.0)

    assert creeps[0].speed == creeps[0].max_speed / 8
    assert creeps[0].hp == 10000.0 - base_damage
    assert creeps[2].speed == creeps[2].max_speed / 8
    assert creeps[2].hp == 10000.0 - base_damage
    assert creeps[3].speed == creeps[3].max_speed / 8
    assert creeps[3].hp == 10000.0 - base_damage
    assert creeps[4].speed == creeps[4].max_speed / 8
    assert creeps[4].hp == 10000.0 - base_damage

    assert creeps[1].speed == 0.5
    assert creeps[1].hp == 10000.0
    assert creeps[5].speed == 1.0
    assert creeps[5].hp == 10000.0
    assert len(tower.shot_segments) == 4


def test_blue2_stuns_fastest_target() -> None:
    state = GameState()
    map_data = _make_map()
    state.bank = 99999
    tower = place_tower(state, map_data, 0, 0, "blue2")
    assert tower is not None
    base_damage = float(tower.damage)

    slow = _make_creep(x=30, y=20, speed=0.5, max_speed=1.0)
    fast = _make_creep(x=35, y=20, speed=1.0, max_speed=1.0)
    state.creeps.extend([slow, fast])

    step_towers(state, map_data, dt_scale=1.0)

    assert fast.speed == -0.2
    assert fast.hp == 10000.0 - base_damage
    assert slow.speed == 0.5
    assert slow.hp == 10000.0
    assert len(tower.shot_segments) == 1
