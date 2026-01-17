# src/vectortd/core/rules/tower_attack.py
from __future__ import annotations

import math
import time

from ..model.entities import Creep, Tower, PulseShot, RocketShot
from ..rng import rand_index

SHOT_FLASH_FRAMES = 2.0
BLUE_SHOT_FRAMES = 5.0
BLUE_SHOT_OPACITY_START = 200
BLUE_RAY_MAX_TARGETS = 4
BLUE_RAY_SPEED_DIVISOR = 1.2
BLUE_RAY_SLOW_DIVISOR = 8.0
BLUE2_STUN_SPEED = -0.2
BLUE2_RETARGET_DELAY = 2.0
RED_RETARGET_DELAY = 2.0
RED_SPLASH_RADIUS = 50.0
RED_SPLASH_RADIUS_SQ = RED_SPLASH_RADIUS * RED_SPLASH_RADIUS
RED_SHOT_FRAMES = 4.0
RED_SHOT_OPACITY = 160
ROCKET_BOUNDS_MIN = 25.0
ROCKET_MAX_SPEED = 3.0
ROCKET_ACCEL = 0.1
SWARM_MAX_SPEED = 4.0
SWARM_ACCEL = 0.2
BOUNCE_RADIUS = 50.0
BOUNCE_RADIUS_SQ = BOUNCE_RADIUS * BOUNCE_RADIUS
PULSE_ALPHA_START = 10.0
PULSE_ALPHA_STEP = 3.0
PULSE_ALPHA_MAX = 100.0
PULSE_BOUNDS_MIN = 25.0


def step_towers(
    state,
    map_data,
    *,
    dt_scale: float = 1.0,
    timing: dict[str, float] | None = None,
    timing_mode: str = "full",
) -> None:
    """
    Tick tour -> cible -> tir -> dégâts.
    """
    if getattr(state, "paused", False):
        return
    if getattr(state, "game_over", False):
        return

    towers = getattr(state, "towers", [])
    if not towers:
        return

    creeps = getattr(state, "creeps", [])
    width = float(getattr(map_data, "width", 0.0))
    height = float(getattr(map_data, "height", 0.0))
    grid = float(getattr(map_data, "grid", 25.0))
    timing_full = timing is not None and timing_mode == "full"
    timing_kind = timing is not None and timing_mode == "tower_kind"
    if timing_full or timing_kind:
        perf_counter = time.perf_counter
    if timing_full:
        pulse_time = 0.0
        rocket_time = 0.0
        timer_time = 0.0
        target_time = 0.0
        fire_time = 0.0
        special_time = 0.0
        tower_count = 0.0
    if timing_kind:
        kind_time_totals: dict[str, float] = {}

    if timing_full:
        t0 = perf_counter()
    _step_pulses(state, creeps, width, height, dt_scale)
    if timing_full:
        pulse_time += perf_counter() - t0

    if timing_full:
        t0 = perf_counter()
    _step_rockets(state, creeps, width, height, dt_scale)
    if timing_full:
        rocket_time += perf_counter() - t0

    for tower in towers:
        if timing_kind:
            kind_start = perf_counter()
        if timing_full:
            tower_count += 1.0
        tick_cooldown = not (tower.kind == "red_spammer" and not creeps)
        if timing_full:
            t0 = perf_counter()
        if (tick_cooldown and tower.cooldown > 0.0) or tower.shot_timer > 0.0:
            _tick_tower_timers(tower, dt_scale, tick_cooldown=tick_cooldown)
        if timing_full:
            timer_time += perf_counter() - t0

        if tower.kind == "red_spammer" and not creeps:
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if tower.cooldown > 0.0:
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if tower.kind in ("blue1", "blue2"):
            if timing_full:
                t0 = perf_counter()
            tower.target = None
            origin_x, origin_y = _tower_center(tower, grid)
            if tower.kind == "blue1":
                _fire_blue_rays(state, tower, creeps, width, height, origin_x, origin_y)
                if tower.rof > 0:
                    tower.cooldown = float(tower.rof)
            else:
                target = _select_blue2_target(tower, creeps, width, height, grid)
                if target is None:
                    tower.cooldown = max(tower.cooldown, BLUE2_RETARGET_DELAY)
                else:
                    _fire_blue_bash(state, tower, target, origin_x, origin_y)
                    if tower.rof > 0:
                        tower.cooldown = float(tower.rof)
            if timing_full:
                special_time += perf_counter() - t0
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if tower.kind == "red_refractor":
            if timing_full:
                t0 = perf_counter()
            origin_x, origin_y = _tower_center(tower, grid)
            _fire_red_refractor(
                state,
                tower,
                creeps,
                width,
                height,
                origin_x,
                origin_y,
                grid,
            )
            if timing_full:
                special_time += perf_counter() - t0
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if tower.kind == "red_spammer":
            if timing_full:
                t0 = perf_counter()
            _fire_red_spammer(state, tower, creeps, width, height, grid)
            if timing_full:
                special_time += perf_counter() - t0
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if tower.kind == "red_rockets":
            if timing_full:
                t0 = perf_counter()
            origin_x, origin_y = _tower_center(tower, grid)
            _fire_red_rockets(state, tower, creeps, width, height, origin_x, origin_y, grid)
            if timing_full:
                special_time += perf_counter() - t0
            if timing_kind:
                kind_key = f"tower_kind_time_{tower.kind}"
                kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                    perf_counter() - kind_start
                )
            continue

        if timing_full:
            t0 = perf_counter()
        target = tower.target
        if target is not None:
            if not getattr(target, "alive", True):
                tower.target = None
                target = None
            elif not _is_target_valid(target, width, height, tower, grid):
                tower.target = None
                tower.cooldown = max(tower.cooldown, float(tower.retarget_delay))
                target = None

        if target is None:
            target = _select_target(tower, creeps, width, height, grid, state)
            if target is None:
                tower.cooldown = max(tower.cooldown, float(tower.retarget_delay))
                if timing_full:
                    target_time += perf_counter() - t0
                if timing_kind:
                    kind_key = f"tower_kind_time_{tower.kind}"
                    kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                        perf_counter() - kind_start
                    )
                continue
            tower.target = target
        if timing_full:
            target_time += perf_counter() - t0

        if timing_full:
            t0 = perf_counter()
        origin_x, origin_y = _tower_center(tower, grid)
        _fire_tower(state, tower, target, origin_x, origin_y)
        if timing_full:
            fire_time += perf_counter() - t0
        if timing_kind:
            kind_key = f"tower_kind_time_{tower.kind}"
            kind_time_totals[kind_key] = kind_time_totals.get(kind_key, 0.0) + (
                perf_counter() - kind_start
            )

    if timing_full:
        timing["tower_pulse_time_total"] = timing.get("tower_pulse_time_total", 0.0) + pulse_time
        timing["tower_rocket_time_total"] = timing.get("tower_rocket_time_total", 0.0) + rocket_time
        timing["tower_timer_time_total"] = timing.get("tower_timer_time_total", 0.0) + timer_time
        timing["tower_target_time_total"] = timing.get("tower_target_time_total", 0.0) + target_time
        timing["tower_fire_time_total"] = timing.get("tower_fire_time_total", 0.0) + fire_time
        timing["tower_special_time_total"] = timing.get("tower_special_time_total", 0.0) + special_time
        timing["tower_count_total"] = timing.get("tower_count_total", 0.0) + tower_count
    if timing_kind:
        for key, value in kind_time_totals.items():
            timing[key] = timing.get(key, 0.0) + value


def _tick_tower_timers(tower: Tower, dt_scale: float, *, tick_cooldown: bool = True) -> None:
    if tick_cooldown and tower.cooldown > 0.0:
        tower.cooldown = max(0.0, tower.cooldown - dt_scale)
    if tower.shot_timer > 0.0:
        tower.shot_timer = max(0.0, tower.shot_timer - dt_scale)
        if tower.kind in ("blue1", "blue2"):
            if tower.shot_timer <= 0.0:
                tower.shot_opacity = 0
            else:
                tower.shot_opacity = int(BLUE_SHOT_OPACITY_START * (tower.shot_timer / BLUE_SHOT_FRAMES))


def _tower_center(tower: Tower, grid: float) -> tuple[float, float]:
    return tower.cell_x + grid * 0.5, tower.cell_y + grid * 0.5


def _is_target_valid(
    target: Creep,
    width: float,
    height: float,
    tower: Tower,
    grid: float,
) -> bool:
    margin = _target_bounds_margin(tower)
    if not _creep_in_bounds(target, width, height, margin=margin):
        return False
    tx, ty = _tower_center(tower, grid)
    return _distance_sq(tx, ty, target.x, target.y) <= float(tower.range) ** 2


def _select_target(
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    grid: float,
    state,
) -> Creep | None:
    mode = str(getattr(tower, "target_mode", "closest") or "closest")
    tx, ty = _tower_center(tower, grid)
    range_sq = float(tower.range) ** 2

    if mode == "random":
        candidates = _creeps_in_range(tower, creeps, width, height, grid)
        if not candidates:
            return None
        return _choose_by_mode(mode, candidates, tower, grid, state)

    qualifier = 0.0
    chosen: Creep | None = None
    for creep in creeps:
        if not _creep_in_bounds(creep, width, height, margin=0.0):
            continue
        dist_sq = _distance_sq(tx, ty, creep.x, creep.y)
        if dist_sq > range_sq:
            continue
        if mode == "fastest":
            value = float(creep.speed)
            if value > qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        elif mode == "hardest":
            value = float(creep.hp)
            if value > qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        elif mode == "weakest":
            value = float(creep.hp)
            if value < qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        else:
            if dist_sq < qualifier or qualifier == 0.0:
                qualifier = dist_sq
                chosen = creep
    return chosen


def _creeps_in_range(
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    grid: float,
) -> list[Creep]:
    tx, ty = _tower_center(tower, grid)
    range_sq = float(tower.range) ** 2
    candidates: list[Creep] = []
    for creep in creeps:
        if not _creep_in_bounds(creep, width, height, margin=0.0):
            continue
        if _distance_sq(tx, ty, creep.x, creep.y) <= range_sq:
            candidates.append(creep)
    return candidates


def _select_blue2_target(
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    grid: float,
) -> Creep | None:
    tx, ty = _tower_center(tower, grid)
    range_sq = float(tower.range) ** 2
    qualifier = 0.0
    fastest: Creep | None = None
    for creep in creeps:
        if not _creep_in_bounds(creep, width, height, margin=0.0):
            continue
        if _distance_sq(tx, ty, creep.x, creep.y) > range_sq:
            continue
        if creep.speed > qualifier or qualifier == 0:
            qualifier = float(creep.speed)
            fastest = creep
    return fastest


def _select_red_target(
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    grid: float,
) -> Creep | None:
    mode = str(getattr(tower, "target_mode", "closest") or "closest")
    if mode not in ("closest", "weakest", "hardest"):
        return None
    tx, ty = _tower_center(tower, grid)
    range_sq = float(tower.range) ** 2
    qualifier = 0.0
    chosen: Creep | None = None
    for creep in creeps:
        if not _creep_in_bounds(creep, width, height, margin=0.0):
            continue
        dist_sq = _distance_sq(tx, ty, creep.x, creep.y)
        if dist_sq > range_sq:
            continue
        if mode == "weakest":
            value = float(creep.hp)
            if value < qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        elif mode == "hardest":
            value = float(creep.hp)
            if value > qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        else:
            if dist_sq < qualifier or qualifier == 0.0:
                qualifier = dist_sq
                chosen = creep
    return chosen


def _creep_in_bounds(creep: Creep, width: float, height: float, *, margin: float = 0.0) -> bool:
    return (0.0 + margin) < creep.x < width and (0.0 + margin) < creep.y < height


def _target_bounds_margin(tower: Tower) -> float:
    if tower.kind == "green2":
        return 25.0
    return 0.0


def _choose_by_mode(
    mode: str,
    candidates: list[Creep],
    tower: Tower,
    grid: float,
    state,
) -> Creep:
    if mode == "random":
        return candidates[rand_index(state, len(candidates))]

    qualifier = 0.0
    chosen: Creep | None = None

    if mode == "fastest":
        for creep in candidates:
            value = float(creep.speed)
            if value > qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        return chosen or candidates[0]

    if mode == "hardest":
        for creep in candidates:
            value = float(creep.hp)
            if value > qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        return chosen or candidates[0]

    if mode == "weakest":
        for creep in candidates:
            value = float(creep.hp)
            if value < qualifier or qualifier == 0.0:
                qualifier = value
                chosen = creep
        return chosen or candidates[0]

    for creep in candidates:
        value = _distance_to_tower_sq(tower, creep, grid)
        if value < qualifier or qualifier == 0.0:
            qualifier = value
            chosen = creep
    return chosen or candidates[0]


def _distance_to_tower_sq(tower: Tower, creep: Creep, grid: float) -> float:
    tx, ty = _tower_center(tower, grid)
    return _distance_sq(tx, ty, creep.x, creep.y)


def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _fire_tower(
    state,
    tower: Tower,
    target: Creep,
    origin_x: float,
    origin_y: float,
) -> None:
    if tower.kind == "green":
        _fire_green_laser(state, tower, target, origin_x, origin_y)
    elif tower.kind == "green2":
        _fire_green_laser_chain(state, tower, target, origin_x, origin_y, bounces=1)
    elif tower.kind == "green3":
        _fire_green_laser_chain(state, tower, target, origin_x, origin_y, bounces=2)
    elif tower.kind == "purple1":
        _spawn_purple_pulses(state, tower, target, origin_x, origin_y, beams=1, slow=False)
    elif tower.kind == "purple2":
        _spawn_purple_pulses(state, tower, target, origin_x, origin_y, beams=2, slow=False)
    elif tower.kind == "purple3":
        _spawn_purple_pulses(state, tower, target, origin_x, origin_y, beams=1, slow=True)
    else:
        _fire_basic(state, tower, target, origin_x, origin_y)

    if tower.rof > 0:
        tower.cooldown = float(tower.rof)


def _fire_basic(state, tower: Tower, target: Creep, origin_x: float, origin_y: float) -> None:
    _record_shot_segments(tower, [(origin_x, origin_y, float(target.x), float(target.y))])
    _apply_damage(state, target, float(tower.damage))


def _fire_green_laser(state, tower: Tower, target: Creep, origin_x: float, origin_y: float) -> None:
    _record_shot_segments(tower, [(origin_x, origin_y, float(target.x), float(target.y))])
    dmg = _green_laser_damage(float(tower.damage), target.type_id)
    _apply_damage(state, target, dmg)


def _fire_red_refractor(
    state,
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
    grid: float,
) -> None:
    tower.cooldown = float(tower.rof)
    target = tower.target
    if target is not None and not getattr(target, "alive", True):
        target = None
        tower.target = None

    if target is not None:
        if not _creep_in_bounds(target, width, height, margin=0.0):
            tower.target = None
            return
        if _distance_sq(origin_x, origin_y, target.x, target.y) > float(tower.range) ** 2:
            tower.target = None
            return
        _apply_red_refractor_hit(
            state,
            tower,
            target,
            creeps,
            width,
            height,
            origin_x,
            origin_y,
        )
        return

    target = _select_red_target(tower, creeps, width, height, grid)
    if target is None:
        tower.cooldown = RED_RETARGET_DELAY
        return
    tower.target = target
    _apply_red_refractor_hit(
        state,
        tower,
        target,
        creeps,
        width,
        height,
        origin_x,
        origin_y,
    )


def _apply_red_refractor_hit(
    state,
    tower: Tower,
    target: Creep,
    creeps: list[Creep],
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
) -> None:
    segments: list[tuple[float, float, float, float]] = [
        (origin_x, origin_y, float(target.x), float(target.y))
    ]
    dmg = _red_damage(float(tower.damage), target.type_id)
    target.hp -= dmg

    kills: list[Creep] = []
    kill_ids: set[int] = set()
    for creep in creeps:
        if creep is target:
            continue
        if not getattr(creep, "alive", True):
            continue
        if not _creep_in_bounds(creep, width, height, margin=ROCKET_BOUNDS_MIN):
            continue
        dx = creep.x - target.x
        dy = creep.y - target.y
        if dx > RED_SPLASH_RADIUS or dx < -RED_SPLASH_RADIUS:
            continue
        if dy > RED_SPLASH_RADIUS or dy < -RED_SPLASH_RADIUS:
            continue
        dist_sq = dx * dx + dy * dy
        if dist_sq > RED_SPLASH_RADIUS_SQ:
            continue
        dist = math.sqrt(dist_sq)
        splash = dmg / 50.0 * (50.0 - dist / 2.0)
        creep.hp -= splash
        if creep.hp <= 0 and id(creep) not in kill_ids:
            kill_ids.add(id(creep))
            kills.append(creep)
        segments.append(
            (
                float(target.x),
                float(target.y),
                float(creep.x),
                float(creep.y),
            )
        )

    if target.hp <= 0 and id(target) not in kill_ids:
        kill_ids.add(id(target))
        kills.append(target)

    for creep in kills:
        _kill_creep(state, creep)

    _record_shot_segments(tower, segments)
    tower.shot_timer = max(tower.shot_timer, RED_SHOT_FRAMES)
    tower.shot_opacity = RED_SHOT_OPACITY


def _fire_red_spammer(
    state,
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    grid: float,
) -> None:
    tower.target = None
    tower.cooldown = float(tower.rof or 4)
    candidates = _creeps_in_range(tower, creeps, width, height, grid)
    if not candidates:
        return
    target = candidates[rand_index(state, len(candidates))]
    cell_size = max(1, int(round(grid)))
    origin_x = float(tower.cell_x + rand_index(state, cell_size))
    origin_y = float(tower.cell_y + rand_index(state, cell_size))
    _spawn_swarm_rocket(state, target, origin_x, origin_y, float(tower.damage))


def _fire_red_rockets(
    state,
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
    grid: float,
) -> None:
    tower.cooldown = float(tower.rof)
    target = tower.target
    if target is not None and not getattr(target, "alive", True):
        target = None
        tower.target = None

    if target is not None:
        if not _creep_in_bounds(target, width, height, margin=0.0):
            tower.target = None
            return
        if _distance_sq(origin_x, origin_y, target.x, target.y) > float(tower.range) ** 2:
            tower.target = None
            return
        _spawn_red_rocket(state, target, origin_x, origin_y, float(tower.damage))
        return

    target = _select_red_target(tower, creeps, width, height, grid)
    if target is None:
        tower.cooldown = RED_RETARGET_DELAY
        return
    tower.target = target
    _spawn_red_rocket(state, target, origin_x, origin_y, float(tower.damage))


def _spawn_red_rocket(
    state,
    target: Creep,
    origin_x: float,
    origin_y: float,
    damage: float,
) -> None:
    rockets = getattr(state, "rockets", None)
    if rockets is None:
        return
    rockets.append(
        RocketShot(
            kind="rocket",
            x=float(origin_x),
            y=float(origin_y),
            target=target,
            speed=0.0,
            damage=float(damage),
        )
    )


def _spawn_swarm_rocket(
    state,
    target: Creep,
    origin_x: float,
    origin_y: float,
    damage: float,
) -> None:
    rockets = getattr(state, "rockets", None)
    if rockets is None:
        return
    rockets.append(
        RocketShot(
            kind="swarm",
            x=float(origin_x),
            y=float(origin_y),
            target=target,
            speed=2.0,
            damage=float(damage),
        )
    )


def _fire_blue_rays(
    state,
    tower: Tower,
    creeps: list[Creep],
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
) -> None:
    range_sq = float(tower.range) ** 2
    segments: list[tuple[float, float, float, float]] = []
    for creep in creeps:
        if len(segments) >= BLUE_RAY_MAX_TARGETS:
            break
        if creep.speed <= float(creep.max_speed) / BLUE_RAY_SPEED_DIVISOR:
            continue
        if not _creep_in_bounds(creep, width, height, margin=0.0):
            continue
        if _distance_sq(origin_x, origin_y, creep.x, creep.y) > range_sq:
            continue
        segments.append((origin_x, origin_y, float(creep.x), float(creep.y)))
        creep.speed = float(creep.max_speed) / BLUE_RAY_SLOW_DIVISOR
        dmg = _blue_ray_damage(float(tower.damage), creep.type_id)
        _apply_damage(state, creep, dmg)
    if segments:
        _record_blue_shot_segments(tower, segments)


def _fire_blue_bash(state, tower: Tower, target: Creep, origin_x: float, origin_y: float) -> None:
    segments = [(origin_x, origin_y, float(target.x), float(target.y))]
    target.speed = BLUE2_STUN_SPEED
    dmg = _blue_ray_damage(float(tower.damage), target.type_id)
    _apply_damage(state, target, dmg)
    _record_blue_shot_segments(tower, segments)


def _record_shot_segments(tower: Tower, segments: list[tuple[float, float, float, float]]) -> None:
    tower.shot_timer = max(tower.shot_timer, SHOT_FLASH_FRAMES)
    tower.shot_segments = segments
    tower.shot_opacity = 200
    if segments:
        _, _, end_x, end_y = segments[-1]
        tower.shot_x = float(end_x)
        tower.shot_y = float(end_y)


def _record_blue_shot_segments(tower: Tower, segments: list[tuple[float, float, float, float]]) -> None:
    tower.shot_timer = max(tower.shot_timer, BLUE_SHOT_FRAMES)
    tower.shot_segments = segments
    tower.shot_opacity = BLUE_SHOT_OPACITY_START
    if segments:
        _, _, end_x, end_y = segments[-1]
        tower.shot_x = float(end_x)
        tower.shot_y = float(end_y)


def _green_laser_damage(base: float, target_type: int) -> float:
    if target_type == 3:
        return base * 1.5
    if target_type == 1:
        return base * 0.5
    return base


def _blue_ray_damage(base: float, target_type: int) -> float:
    if target_type == 2:
        return base * 1.5
    # AS code checks this.targ.Type for purple, which never matches in practice.
    return base


def _red_damage(base: float, target_type: int) -> float:
    if target_type == 1:
        return base * 1.5
    if target_type == 3:
        return base * 0.5
    return base


def _purple_pulse_damage(base: float, target_type: int) -> float:
    if target_type == 5:
        return base * 1.5
    if target_type == 2:
        return base * 0.5
    return base


def _fire_green_laser_chain(
    state,
    tower: Tower,
    target: Creep,
    origin_x: float,
    origin_y: float,
    *,
    bounces: int,
) -> None:
    segments: list[tuple[float, float, float, float]] = [
        (origin_x, origin_y, float(target.x), float(target.y))
    ]
    dmg = _green_laser_damage(float(tower.damage), target.type_id)
    target.hp -= dmg
    if target.hp <= 0:
        _kill_creep(state, target)
        _record_shot_segments(tower, segments)
        return

    prev = target
    hit_ids: set[int] = {id(target)}
    for _ in range(bounces):
        bounce = _find_bounce_target(state, prev, hit_ids)
        if bounce is None:
            break
        segments.append((float(prev.x), float(prev.y), float(bounce.x), float(bounce.y)))
        bounce.hp -= dmg
        if bounce.hp <= 0:
            _kill_creep(state, bounce)
            break
        hit_ids.add(id(bounce))
        prev = bounce

    _record_shot_segments(tower, segments)


def _find_bounce_target(
    state,
    origin: Creep,
    hit_ids: set[int],
) -> Creep | None:
    creeps = getattr(state, "creeps", [])
    ox = float(origin.x)
    oy = float(origin.y)
    for creep in creeps:
        if id(creep) in hit_ids:
            continue
        if _distance_sq(ox, oy, creep.x, creep.y) <= BOUNCE_RADIUS_SQ:
            return creep
    return None


def _spawn_purple_pulses(
    state,
    tower: Tower,
    target: Creep,
    origin_x: float,
    origin_y: float,
    *,
    beams: int,
    slow: bool,
) -> None:
    pulses = getattr(state, "pulses", None)
    if pulses is None:
        return
    offsets = [0.0]
    if beams == 2:
        offsets = [-2.0, 2.0]
    for offset in offsets:
        pulses.append(
            PulseShot(
                tower=tower,
                target=target,
                from_x=float(origin_x + offset),
                from_y=float(origin_y),
                damage=float(tower.damage),
                alpha=PULSE_ALPHA_START,
                slow=slow,
            )
        )


def _step_rockets(
    state,
    creeps: list[Creep],
    width: float,
    height: float,
    dt_scale: float,
) -> None:
    rockets = getattr(state, "rockets", [])
    if not rockets:
        return
    keep: list[RocketShot] = []

    bounds_25: list[bool] | None = None
    creeps_snapshot: list[Creep] | None = None

    for rocket in rockets:
        target = rocket.target
        if target is None or not getattr(target, "alive", True):
            target = None

        if rocket.kind == "rocket":
            if target is None:
                if bounds_25 is None:
                    creeps_snapshot = list(creeps)
                    bounds_25 = [False] * len(creeps_snapshot)
                    for idx, creep in enumerate(creeps_snapshot):
                        x = float(creep.x)
                        y = float(creep.y)
                        bounds_25[idx] = (ROCKET_BOUNDS_MIN < x < width) and (
                            ROCKET_BOUNDS_MIN < y < height
                        )
                target = _find_closest_creep(
                    rocket.x,
                    rocket.y,
                    creeps_snapshot,
                    width,
                    height,
                    in_bounds_25=bounds_25,
                )
                if target is None:
                    continue
            if not _creep_in_bounds(target, width, height, margin=ROCKET_BOUNDS_MIN):
                continue
            rocket.speed = min(ROCKET_MAX_SPEED, rocket.speed + ROCKET_ACCEL * dt_scale)
            if _rocket_hits_target(rocket, target, dt_scale):
                dmg = _red_damage(rocket.damage, target.type_id)
                _apply_damage(state, target, dmg)
                continue
        else:
            if target is None:
                continue
            if not _creep_in_bounds(target, width, height, margin=ROCKET_BOUNDS_MIN):
                continue
            rocket.speed = min(SWARM_MAX_SPEED, rocket.speed + SWARM_ACCEL * dt_scale)
            if _rocket_hits_target(rocket, target, dt_scale):
                dmg = _red_damage(rocket.damage, target.type_id)
                _apply_damage(state, target, dmg)
                continue

        rocket.target = target
        keep.append(rocket)

    state.rockets = keep


def _find_closest_creep(
    x: float,
    y: float,
    creeps: list[Creep],
    width: float,
    height: float,
    *,
    in_bounds_25: list[bool] | None = None,
) -> Creep | None:
    qualifier = 0.0
    chosen: Creep | None = None
    for idx, creep in enumerate(creeps):
        if not getattr(creep, "alive", True):
            continue
        if in_bounds_25 is None:
            if not _creep_in_bounds(creep, width, height, margin=ROCKET_BOUNDS_MIN):
                continue
        else:
            if idx >= len(in_bounds_25) or not in_bounds_25[idx]:
                continue
        dist_sq = _distance_sq(x, y, creep.x, creep.y)
        if dist_sq < qualifier or qualifier == 0.0:
            qualifier = dist_sq
            chosen = creep
    return chosen


def _rocket_hits_target(rocket: RocketShot, target: Creep, dt_scale: float) -> bool:
    dx = target.x - rocket.x
    dy = target.y - rocket.y
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return True
    step = rocket.speed * dt_scale
    if step <= 0.0:
        return False
    rocket.x += (dx / dist) * step
    rocket.y += (dy / dist) * step
    new_dist = math.hypot(target.x - rocket.x, target.y - rocket.y)
    return new_dist <= step


def _step_pulses(state, creeps: list[Creep], width: float, height: float, dt_scale: float) -> None:
    pulses = getattr(state, "pulses", [])
    if not pulses:
        return

    tower_segments: dict[int, list[tuple[float, float, float, float]]] = {}
    tower_opacity: dict[int, int] = {}
    tower_refs: dict[int, Tower] = {}
    keep: list[PulseShot] = []

    for pulse in pulses:
        target = pulse.target
        if not getattr(target, "alive", True):
            continue
        if not _pulse_in_bounds(target, width, height):
            continue

        pulse.alpha = min(PULSE_ALPHA_MAX, pulse.alpha + PULSE_ALPHA_STEP * dt_scale)
        if pulse.slow:
            target.speed = target.max_speed / 100.0 * (100.0 - pulse.alpha)
        if pulse.alpha >= PULSE_ALPHA_MAX:
            dmg = _purple_pulse_damage(pulse.damage, target.type_id)
            _apply_damage(state, target, dmg)
            continue

        keep.append(pulse)
        tower = pulse.tower
        tower_id = id(tower)
        tower_refs[tower_id] = tower
        tower_segments.setdefault(tower_id, []).append(
            (pulse.from_x, pulse.from_y, float(target.x), float(target.y))
        )
        opacity = int(30 + (pulse.alpha / PULSE_ALPHA_MAX) * 200)
        tower_opacity[tower_id] = max(opacity, tower_opacity.get(tower_id, 0))

    state.pulses = keep

    for tower_id, segments in tower_segments.items():
        tower = tower_refs.get(tower_id)
        if tower is None:
            continue
        tower.shot_segments = segments
        tower.shot_timer = max(tower.shot_timer, 1.0)
        tower.shot_opacity = tower_opacity.get(tower_id, 200)


def _pulse_in_bounds(target: Creep, width: float, height: float) -> bool:
    return PULSE_BOUNDS_MIN < target.x < width and PULSE_BOUNDS_MIN < target.y < height


def _apply_damage(state, target: Creep, amount: float) -> None:
    target.hp -= amount
    if target.hp > 0:
        return
    _kill_creep(state, target)


def _kill_creep(state, target: Creep) -> None:
    creeps = getattr(state, "creeps", [])
    target.alive = False
    try:
        creeps.remove(target)
    except ValueError:
        return
    bank = getattr(state, "bank", None)
    if bank is not None:
        state.bank = int(bank) + int(target.worth)
    score = getattr(state, "score", None)
    if score is not None:
        state.score = int(score) + int(target.worth)
    if getattr(target, "type_id", None) == 6:
        ups = getattr(state, "ups", None)
        if ups is not None:
            state.ups = int(ups) + 1

