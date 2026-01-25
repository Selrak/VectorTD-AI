from types import SimpleNamespace

import pytest

from vectortd.core.model.entities import Creep, Tower
from vectortd.core.model.state import GameState
from vectortd.core.model.towers import get_tower_def
from vectortd.core.rules.buffs import recompute_buffs
from vectortd.core.rules.placement import place_tower
from vectortd.core.rules.tower_attack import step_towers


def _make_tower(kind: str, cell_x: int, cell_y: int) -> Tower:
    tower_def = get_tower_def(kind)
    return Tower(
        cell_x=int(cell_x),
        cell_y=int(cell_y),
        kind=tower_def.kind,
        title=tower_def.title,
        level=tower_def.level,
        cost=tower_def.cost,
        range=tower_def.range,
        damage=tower_def.damage,
        description=tower_def.description,
        base_cost=tower_def.base_cost,
        base_range=tower_def.base_range,
        base_damage=tower_def.base_damage,
        target_mode=tower_def.target_mode,
        rof=tower_def.rof,
        cooldown=0.0,
        retarget_delay=(10 if tower_def.rof == 0 else tower_def.rof),
        target=None,
        shot_timer=0.0,
        shot_x=0.0,
        shot_y=0.0,
        shot_segments=[],
        shot_opacity=200,
    )


def _make_creep(
    *,
    x: float,
    y: float,
    hp: float = 1000.0,
    type_id: int = 2,
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
        speed=1.0,
        max_speed=1.0,
        hp=float(hp),
        maxhp=float(hp),
    )


def test_recompute_buffs_stacks_and_strict_radius() -> None:
    state = GameState()
    map_data = SimpleNamespace(grid=25)

    base = _make_tower("green", 0, 0)
    buff_d1 = _make_tower("buffD", 75, 0)
    buff_d2 = _make_tower("buffD", 0, 75)
    buff_r_inside = _make_tower("buffR", 75, 25)
    buff_r_edge = _make_tower("buffR", 100, 0)

    state.towers.extend([base, buff_d1, buff_d2, buff_r_inside, buff_r_edge])

    recompute_buffs(state, map_data)

    assert base.damage_buff_pct == 50
    assert base.range_buff_pct == 25
    assert base.buffed_damage == pytest.approx(base.damage * 1.5)
    assert base.buffed_range == pytest.approx(base.range * 1.25)


def test_buff_damage_applies_to_attack() -> None:
    state = GameState()
    state.bank = 99999
    state.ups = 1
    map_data = SimpleNamespace(width=300, height=300, grid=25, paths=[])

    tower = place_tower(state, map_data, 0, 0, "green")
    assert tower is not None
    buff = place_tower(state, map_data, 75, 0, "buffD")
    assert buff is not None

    creep = _make_creep(x=20, y=20, hp=1000.0, type_id=2)
    state.creeps.append(creep)

    start_hp = creep.hp
    step_towers(state, map_data, dt_scale=1.0)

    expected_damage = tower.damage * 1.25
    assert creep.hp == pytest.approx(start_hp - expected_damage)
