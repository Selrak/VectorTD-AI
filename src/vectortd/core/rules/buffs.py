from __future__ import annotations

from ..model.towers import is_buff_tower

BUFF_RADIUS = 100.0
BUFF_PCT = 25


def recompute_buffs(state, map_data) -> None:
    towers = list(getattr(state, "towers", []) or [])
    if not towers:
        return
    grid = float(getattr(map_data, "grid", 25.0) or 25.0)

    normal_towers = [t for t in towers if not is_buff_tower(getattr(t, "kind", ""))]
    buff_towers = [t for t in towers if is_buff_tower(getattr(t, "kind", ""))]

    for tower in normal_towers:
        tower.damage_buff_pct = 0
        tower.range_buff_pct = 0

    for tower in buff_towers:
        tower.damage_buff_pct = 0
        tower.range_buff_pct = 0

    if normal_towers and buff_towers:
        radius_sq = BUFF_RADIUS * BUFF_RADIUS
        for buff in buff_towers:
            bx, by = _tower_center(buff, grid)
            kind = getattr(buff, "kind", "")
            for tower in normal_towers:
                tx, ty = _tower_center(tower, grid)
                dx = bx - tx
                dy = by - ty
                if (dx * dx + dy * dy) >= radius_sq:
                    continue
                if kind == "buffD":
                    tower.damage_buff_pct += BUFF_PCT
                elif kind == "buffR":
                    tower.range_buff_pct += BUFF_PCT

    for tower in normal_towers:
        base_range = float(getattr(tower, "range", 0.0))
        base_damage = float(getattr(tower, "damage", 0.0))
        tower.buffed_range = base_range + base_range * (tower.range_buff_pct / 100.0)
        tower.buffed_damage = base_damage + base_damage * (tower.damage_buff_pct / 100.0)

    for tower in buff_towers:
        tower.buffed_range = float(getattr(tower, "range", 0.0))
        tower.buffed_damage = float(getattr(tower, "damage", 0.0))


def buy_interest(state) -> bool:
    if getattr(state, "paused", False):
        return False
    ups = getattr(state, "ups", None)
    if ups is None or int(ups) < 1:
        return False
    state.interest = int(getattr(state, "interest", 0)) + 3
    state.ups = int(ups) - 1
    return True


def buy_emergency_lives(state) -> bool:
    if getattr(state, "paused", False):
        return False
    ups = getattr(state, "ups", None)
    if ups is None or int(ups) < 1:
        return False
    state.lives = int(getattr(state, "lives", 0)) + 5
    state.ups = int(ups) - 1
    return True


def _tower_center(tower, grid: float) -> tuple[float, float]:
    return (
        float(getattr(tower, "cell_x", 0)) + grid * 0.5,
        float(getattr(tower, "cell_y", 0)) + grid * 0.5,
    )
