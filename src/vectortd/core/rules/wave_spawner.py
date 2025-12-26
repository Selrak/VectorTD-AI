# src/vectortd/core/rules/wave_spawner.py
from __future__ import annotations

import math
from typing import Literal

# Le spawner manipule state/map mais reste découplé de la GUI.
# Il ne dépend que des champs utilisés (duck-typing).

# Table de vagues (depuis le SWF / spec).
LEVELS: list[int] = [
    2, 1, 2, 3, 7, 4, 2, 5, 2, 7,
    2, 3, 2, 4, 7, 5, 2, 1, 2, 7,
    2, 4, 2, 5, 7, 1, 2, 3, 2, 7,
    4, 2, 5, 2, 7, 5, 2, 1, 2, 8
]


def _int(x: float) -> int:
    # Flash int() tronque vers 0.
    return int(x)


def _spawn_offset(spawn_dir: Literal["up", "down", "left", "right"]) -> tuple[int, int]:
    if spawn_dir == "up":
        return (0, -20)
    if spawn_dir == "down":
        return (0, +20)
    if spawn_dir == "left":
        return (-20, 0)
    if spawn_dir == "right":
        return (+20, 0)
    raise ValueError(f"Invalid spawn_dir={spawn_dir!r}")


def start_next_wave(state, map_data) -> None:
    """
    Équivalent de wave() du SWF, version minimale et déterministe.

    Effets importants reproduits :
    - level++
    - intérêts : bank/score
    - spawn sur chaque chemin : 14 creeps par lane
    - “bonus creep” toutes les bonusEvery vagues : dernier creep de la dernière lane => type 6
    - difficulté : baseHP augmente, baseWorth++
    """
    if getattr(state, "paused", False):
        return
    if getattr(state, "game_over", False):
        return

    state.level += 1

    # intérêt (avant spawn)
    bank_gain = _int(state.bank / 100 * state.interest)
    state.bank += bank_gain
    score_gain = _int(state.bank / 100 * state.interest)
    state.score += score_gain

    # compteur utilisé par la vague “mixée”
    state.cc = 1

    if state.level > len(LEVELS):
        state.game_over = True
        return

    wave_type = LEVELS[state.level - 1]  # Type = levels[level-1]

    dx, dy = _spawn_offset(map_data.spawn_dir)

    for lane_idx, path in enumerate(map_data.paths):
        # point de base (marker du début du path)
        start_marker = path[0]
        x0, y0 = map_data.marker_xy(start_marker)

        x = x0
        y = y0

        for v in range(1, 15):  # v = 1..14
            x += dx
            y += dy
            _waveB(state, map_data, x, y, path, wave_type, v=v, lane_idx=lane_idx)

    # fin de vague : difficulté
    if map_data.level_index < 5:
        state.base_hp += _int(state.base_hp / 6)
    else:
        state.base_hp += _int(state.base_hp / 5)
    state.base_worth += 1


def dev_spawn_wave(state, map_data, *, creeps_per_lane: int = 3, wave_type: int = 1) -> None:
    """
    Compatibility wrapper for the real wave() logic from the SWF.
    creeps_per_lane/wave_type are ignored to keep the API stable.
    """
    start_next_wave(state, map_data)


def _waveB(state, map_data, x: float, y: float, path: list[int], wave_type: int, *, v: int, lane_idx: int) -> None:
    """
    Bonus creep (équivalent waveB()).
    """
    t = wave_type

    # dernier creep de la vague : v==14, dernière lane, et level multiple de bonusEvery
    if (
        v == 14
        and (state.level % state.bonus_every) == 0
        and lane_idx == (len(map_data.paths) - 1)
    ):
        t = 6

    bHP = state.base_hp
    if t == 6 or t == 7:
        bHP = state.base_hp * 1.5

    _spawn(state, map_data, x, y, path, t, bHP=bHP)


def _spawn(state, map_data, x: float, y: float, path: list[int], t: int, *, bHP: float) -> None:
    """
    Équivalent de spawn(X,Y,Path,Type).

    Cas Type==8 : vague “mixée”.
    """
    # vague mixée
    if t == 8:
        # t = int(cc/5)+1 ; si t==6 -> 7
        t2 = int(state.cc / 5) + 1
        if t2 == 6:
            t2 = 7
        t = t2

    # vitesse
    speed = 2.0 if t == 4 else 1.0

    # HP
    hp = bHP if t != 6 else (bHP * 4)

    # target initiale : marker du 1er point du path
    targ_marker = path[0]
    tx, ty = map_data.marker_xy(targ_marker)

    # direction initiale (normalisée)
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        xval, yval = 0.0, 0.0
    else:
        xval, yval = dx / dist, dy / dist

    # création du creep (le type est défini dans core.engine.Creep)
    CreepCls = None
    try:
        # import local et tardif pour éviter dépendances circulaires
        from ..engine import Creep as CreepCls  # type: ignore
    except Exception:
        pass

    if CreepCls is None:
        raise RuntimeError("Creep class not found (expected core.engine.Creep)")

    c = CreepCls(
        x=float(x),
        y=float(y),
        type_id=int(t),
        path=list(path),
        path_point=0,
        targ_x=float(tx),
        targ_y=float(ty),
        xval=float(xval),
        yval=float(yval),
        worth=int(state.base_worth),
        speed=float(speed),
        max_speed=float(speed),
        hp=float(hp),
        maxhp=float(hp),
    )

    state.creeps.append(c)
    state.cc += 1
