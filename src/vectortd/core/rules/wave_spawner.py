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

CREEP_NAMES: dict[int, str] = {
    1: "Red Shredder",
    2: "Blue Spinner",
    3: "Green Flyer",
    4: "Yellow Sprinter",
    5: "Big Purple Box",
    6: "Bonus",
    7: "Hard Grey + Bonus",
    8: "All types",
}


def creep_name(type_id: int) -> str:
    return CREEP_NAMES.get(int(type_id), f"Type {type_id}")


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
    state.last_wave_type = int(wave_type)
    state.last_wave_hp = int(state.base_hp)

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


def wave_display_info(state, map_data) -> dict[str, dict[str, int | str | bool] | None]:
    """
    Infos UI/observation : vague courante + suivante (inspiré du SWF).
    """
    level = int(getattr(state, "level", 0))
    base_hp = int(getattr(state, "base_hp", 0))
    map_level = int(getattr(map_data, "level_index", 1))

    current = None
    if level > 0 and getattr(state, "last_wave_type", None) is not None:
        cur_type = int(state.last_wave_type)
        cur_hp = int(getattr(state, "last_wave_hp", base_hp))
        current = {
            "type": cur_type,
            "name": creep_name(cur_type),
            "hp": cur_hp,
            "show_sprite": True,
        }

    next_info = None
    if level < len(LEVELS):
        next_type = int(LEVELS[level])
        if level == 0:
            next_hp = base_hp
        else:
            if map_level < 5:
                next_hp = base_hp + _int(base_hp / 5)
            else:
                next_hp = base_hp + _int(base_hp / 4)
        if next_type == 6:
            next_hp = int(next_hp * 1.5)
        next_name = creep_name(next_type)
        show_sprite = True
        if level == (len(LEVELS) - 1):
            next_name = "???"
            show_sprite = False
        next_info = {
            "type": next_type,
            "name": next_name,
            "hp": int(next_hp),
            "show_sprite": show_sprite,
        }

    return {"current": current, "next": next_info}


def maybe_auto_next_wave(state, map_data) -> bool:
    """
    Équivalent de l'autoLevel SWF :
    - si plus de creeps et dernière vague atteinte => game_over
    - si plus de creeps et auto_level actif => wave()
    """
    if getattr(state, "paused", False):
        return False
    if getattr(state, "game_over", False):
        return False

    creeps = getattr(state, "creeps", [])
    if len(creeps) != 0:
        return False

    # Fin des vagues (équivalent creepArray vide et level==40).
    if state.level >= len(LEVELS):
        state.game_over = True
        state.game_won = True
        return True

    if getattr(state, "auto_level", False):
        start_next_wave(state, map_data)
        return True

    return False


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

    # création du creep (type canonique défini dans core.model.entities.Creep)
    CreepCls = None
    try:
        # import local et tardif pour éviter dépendances circulaires
        from ..model.entities import Creep as CreepCls  # type: ignore
    except Exception:
        pass

    if CreepCls is None:
        raise RuntimeError("Creep class not found (expected core.model.entities.Creep)")

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
