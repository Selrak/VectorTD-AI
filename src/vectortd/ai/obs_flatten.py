from __future__ import annotations

from typing import Any


SCALAR_KEYS = (
    "bank_norm",
    "lives_norm",
    "score_norm",
    "wave_norm",
    "interest_norm",
    "ups_norm",
    "base_hp_norm",
    "base_worth_norm",
    "tower_count_norm",
    "wave_current_present",
    "wave_current_type_norm",
    "wave_current_hp_norm",
    "wave_next_present",
    "wave_next_type_norm",
    "wave_next_hp_norm",
)


def flatten_observation(obs: dict[str, Any], *, max_towers: int, slot_size: int) -> list[float]:
    values: list[float] = [float(obs.get(key, 0.0) or 0.0) for key in SCALAR_KEYS]
    tower_slots = obs.get("tower_slots", []) or []
    empty_slot = [0.0] * slot_size
    for idx in range(max_towers):
        slot = tower_slots[idx] if idx < len(tower_slots) else empty_slot
        values.extend(float(value) for value in slot)
    place_candidates = obs.get("place_candidates")
    if place_candidates:
        for candidate in place_candidates:
            values.extend(float(value) for value in candidate)
    return values
