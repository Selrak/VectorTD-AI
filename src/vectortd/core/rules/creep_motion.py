# src/vectortd/core/rules/creep_motion.py
from __future__ import annotations

import math
import time


def step_creeps(state, map_data, *, dt_scale: float = 1.0, timing: dict[str, float] | None = None) -> None:
    """
    Équivalent du tick original des creeps.

    dt_scale=1.0 correspond à “un frame Flash”.
    Si le moteur fait du fixed-step à 60 fps, dt_scale reste 1.0.

    Particularité importante reproduite :
    - quand un creep atteint la fin de son path :
      lives-- ; score -= worth*2 (clamp >=0) ; le creep "boucle" et repart.
    """
    if getattr(state, "paused", False):
        return
    if getattr(state, "game_over", False):
        return

    creeps = getattr(state, "creeps", [])
    if not creeps:
        return

    if timing is None:
        for c in creeps:
            # Avancement
            c.x += c.xval * c.speed * dt_scale
            c.y += c.yval * c.speed * dt_scale

            # Accélération légère vers max_speed
            if c.speed < c.max_speed:
                c.speed = min(c.max_speed, c.speed + 0.01 * dt_scale)

            # Arrivée sur cible (boîte autour du point)
            if abs(c.x - c.targ_x) < (c.speed * dt_scale) and abs(c.y - c.targ_y) < (c.speed * dt_scale):
                # snap
                c.x = c.targ_x
                c.y = c.targ_y
                c.path_point += 1

                # Fin du chemin -> boucle
                if c.path_point >= len(c.path):
                    state.lives -= 1
                    state.score -= int(c.worth) * 2
                    if state.score < 0:
                        state.score = 0
                    if state.lives <= 0:
                        state.game_over = True

                    # reset : repart au début, et vise le 2e point (pathPoint=1)
                    if len(c.path) >= 2:
                        start_x, start_y = map_data.marker_xy(c.path[0])
                        next_x, next_y = map_data.marker_xy(c.path[1])

                        c.x = float(start_x)
                        c.y = float(start_y)
                        c.path_point = 1

                        c.targ_x = float(next_x)
                        c.targ_y = float(next_y)

                        _recompute_dir(c)
                    else:
                        # cas path dégénéré : rester immobile
                        c.path_point = 0
                        c.targ_x, c.targ_y = c.x, c.y
                        c.xval, c.yval = 0.0, 0.0

                else:
                    # Point suivant
                    next_marker = c.path[c.path_point]
                    nx, ny = map_data.marker_xy(next_marker)
                    c.targ_x = float(nx)
                    c.targ_y = float(ny)
                    _recompute_dir(c)
        return

    perf_counter = time.perf_counter
    move_time = 0.0
    path_time = 0.0
    creep_count = 0.0

    for c in creeps:
        creep_count += 1.0
        t0 = perf_counter()
        # Avancement
        c.x += c.xval * c.speed * dt_scale
        c.y += c.yval * c.speed * dt_scale

        # Accélération légère vers max_speed
        if c.speed < c.max_speed:
            c.speed = min(c.max_speed, c.speed + 0.01 * dt_scale)
        move_time += perf_counter() - t0

        # Arrivée sur cible (boîte autour du point)
        if abs(c.x - c.targ_x) < (c.speed * dt_scale) and abs(c.y - c.targ_y) < (c.speed * dt_scale):
            t1 = perf_counter()
            # snap
            c.x = c.targ_x
            c.y = c.targ_y
            c.path_point += 1

            # Fin du chemin -> boucle
            if c.path_point >= len(c.path):
                state.lives -= 1
                state.score -= int(c.worth) * 2
                if state.score < 0:
                    state.score = 0
                if state.lives <= 0:
                    state.game_over = True

                # reset : repart au début, et vise le 2e point (pathPoint=1)
                if len(c.path) >= 2:
                    start_x, start_y = map_data.marker_xy(c.path[0])
                    next_x, next_y = map_data.marker_xy(c.path[1])

                    c.x = float(start_x)
                    c.y = float(start_y)
                    c.path_point = 1

                    c.targ_x = float(next_x)
                    c.targ_y = float(next_y)

                    _recompute_dir(c)
                else:
                    # cas path dégénéré : rester immobile
                    c.path_point = 0
                    c.targ_x, c.targ_y = c.x, c.y
                    c.xval, c.yval = 0.0, 0.0

            else:
                # Point suivant
                next_marker = c.path[c.path_point]
                nx, ny = map_data.marker_xy(next_marker)
                c.targ_x = float(nx)
                c.targ_y = float(ny)
                _recompute_dir(c)
            path_time += perf_counter() - t1

    timing["creep_move_time_total"] = timing.get("creep_move_time_total", 0.0) + move_time
    timing["creep_path_time_total"] = timing.get("creep_path_time_total", 0.0) + path_time
    timing["creep_count_total"] = timing.get("creep_count_total", 0.0) + creep_count


def _recompute_dir(c) -> None:
    dx = c.targ_x - c.x
    dy = c.targ_y - c.y
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        c.xval, c.yval = 0.0, 0.0
    else:
        c.xval, c.yval = dx / dist, dy / dist
