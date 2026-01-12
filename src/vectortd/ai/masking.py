from __future__ import annotations

from vectortd.core.rules.wave_spawner import LEVELS, wave_display_info

from .actions import ActionSpaceSpec, _log_cap_exceeded, get_tower_slots


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
