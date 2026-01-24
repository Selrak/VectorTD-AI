from __future__ import annotations

from vectortd.core.rules.wave_spawner import LEVELS, wave_display_info

from .actions import ActionSpaceSpec, _log_cap_exceeded, get_tower_slots

_CELL_DISTANCE_CACHE: dict[int, list[int]] = {}
_PATH_SEGMENTS_CACHE: dict[int, list[tuple[float, float, float, float]]] = {}


def _path_segments(map_data) -> list[tuple[float, float, float, float]]:
    cache_key = id(map_data)
    cached = _PATH_SEGMENTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    segments: list[tuple[float, float, float, float]] = []
    for path in getattr(map_data, "paths", []):
        for a, b in zip(path[:-1], path[1:]):
            try:
                x1, y1 = map_data.marker_xy(a)
                x2, y2 = map_data.marker_xy(b)
            except Exception:
                continue
            segments.append((float(x1), float(y1), float(x2), float(y2)))
    _PATH_SEGMENTS_CACHE[cache_key] = segments
    return segments


def _point_segment_distance_sq(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = x1 + t * dx
    cy = y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


def _sorted_cell_indices_by_path_distance(map_data, spec: ActionSpaceSpec) -> list[int]:
    cache_key = id(spec)
    cached = _CELL_DISTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    segments = _path_segments(map_data)
    cell_positions = spec.cell_positions
    if not segments or not cell_positions:
        indices = list(range(len(cell_positions)))
        _CELL_DISTANCE_CACHE[cache_key] = indices
        return indices
    grid = int(getattr(map_data, "grid", 25))
    scored: list[tuple[float, int]] = []
    for idx, (cell_x, cell_y) in enumerate(cell_positions):
        cx = float(cell_x + grid * 0.5)
        cy = float(cell_y + grid * 0.5)
        best = None
        for x1, y1, x2, y2 in segments:
            dist_sq = _point_segment_distance_sq(cx, cy, x1, y1, x2, y2)
            if best is None or dist_sq < best:
                best = dist_sq
        scored.append((best if best is not None else 0.0, idx))
    scored.sort(key=lambda item: item[0])
    indices = [idx for _, idx in scored]
    _CELL_DISTANCE_CACHE[cache_key] = indices
    return indices


def _can_start_wave(state, map_data) -> bool:
    if getattr(state, "game_over", False):
        return False
    if getattr(state, "paused", False):
        return False
    creeps = getattr(state, "creeps", []) or []
    if creeps:
        return False
    if getattr(state, "level", 0) >= len(LEVELS):
        return False
    wave_info = wave_display_info(state, map_data)
    return wave_info.get("next") is not None


def compute_action_mask(
    state,
    engine,
    map_data,
    spec: ActionSpaceSpec,
    *,
    phase: str = "BUILD",
    place_cell_top_k: int | None = None,
) -> list[bool]:
    use_numpy = False
    try:
        import numpy as np  # type: ignore

        use_numpy = True
    except Exception:
        np = None  # type: ignore

    if spec.num_actions <= 0:
        return [False] * spec.num_actions

    if use_numpy:
        mask = np.zeros(spec.num_actions, dtype=bool)
    else:
        mask = [False] * spec.num_actions

    mask[spec.offsets.noop] = True

    if phase != "BUILD" or getattr(state, "game_over", False):
        return mask

    if _can_start_wave(state, map_data):
        mask[spec.offsets.start_wave] = True

    towers = getattr(state, "towers", []) or []
    if len(towers) > spec.max_towers:
        _log_cap_exceeded("towers", len(towers), spec.max_towers, context=spec.map_name)

    can_build = len(towers) < spec.max_towers
    if can_build and spec.place_count > 0 and spec.cell_positions:
        cell_count = len(spec.cell_positions)
        if use_numpy:
            occupied = np.zeros(cell_count, dtype=bool)
        else:
            occupied = [False] * cell_count
        cell_index = spec.cell_index
        for tower in towers:
            cell_x = int(getattr(tower, "cell_x", -1))
            cell_y = int(getattr(tower, "cell_y", -1))
            idx = cell_index.get((cell_x, cell_y))
            if idx is not None:
                occupied[idx] = True

        if use_numpy:
            available = ~occupied
        else:
            available = [not is_taken for is_taken in occupied]

        candidates_mask = None
        if place_cell_top_k is not None:
            k = int(place_cell_top_k)
            if k < cell_count:
                candidates = _sorted_cell_indices_by_path_distance(map_data, spec)[: max(k, 0)]
                if use_numpy:
                    candidates_mask = np.zeros(cell_count, dtype=bool)
                    if candidates:
                        candidates_mask[np.asarray(candidates, dtype=int)] = True
                else:
                    candidates_mask = [False] * cell_count
                    for idx in candidates:
                        candidates_mask[idx] = True
        if candidates_mask is not None:
            if use_numpy:
                available = available & candidates_mask
            else:
                available = [is_free and candidates_mask[idx] for idx, is_free in enumerate(available)]

        bank = getattr(state, "bank", None)
        if bank is not None:
            bank_value = int(bank)
            for tower_type, cost in enumerate(spec.tower_costs):
                if bank_value < cost:
                    continue
                base = spec.offsets.place + tower_type * cell_count
                if use_numpy:
                    mask[base: base + cell_count] = available
                else:
                    for cell_idx, is_free in enumerate(available):
                        if is_free:
                            mask[base + cell_idx] = True

    tower_slots = get_tower_slots(state, spec.max_towers)
    bank_value = getattr(state, "bank", None)
    bank_value = int(bank_value) if bank_value is not None else None
    for slot_idx, tower in enumerate(tower_slots):
        if tower is None:
            continue
        if bank_value is not None:
            if int(getattr(tower, "level", 0)) < 10:
                upgrade_cost = int(getattr(tower, "base_cost", 0) / 2)
                if bank_value >= upgrade_cost:
                    mask[spec.offsets.upgrade + slot_idx] = True
        mask[spec.offsets.sell + slot_idx] = True

        if spec.set_mode_count > 0:
            mode_mask = spec.kind_to_mode_mask.get(str(getattr(tower, "kind", "")))
            if mode_mask is None:
                continue
            base = spec.offsets.set_mode + slot_idx * len(spec.target_modes)
            for mode_idx, supported in enumerate(mode_mask):
                if supported:
                    mask[base + mode_idx] = True

    if use_numpy:
        return mask
    return mask
