from __future__ import annotations

from typing import Any
import math

from vectortd.core.model.towers import list_tower_defs
from vectortd.core.rules.wave_spawner import LEVELS, wave_display_info

from .actions import ActionSpaceSpec, sorted_towers


MAX_LEVEL = 10
MAX_LIVES = 20
INTEREST_SCALE = 10.0
BASE_WORTH_SCALE = 50.0
HP_SCALE = 1_000_000.0
SCORE_SCALE_MULTIPLIER = 10.0


def _log_norm(value: int | float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0.0, float(value))) / math.log1p(scale))


def _max_tower_stats(max_level: int) -> tuple[float, float, float, float]:
    max_upgrades = max(0, max_level - 1)
    max_cost = 1.0
    max_range = 1.0
    max_damage = 1.0
    max_rof = 1.0
    for tower_def in list_tower_defs():
        base_cost = float(getattr(tower_def, "base_cost", 0))
        base_range = float(getattr(tower_def, "base_range", 0))
        base_damage = float(getattr(tower_def, "base_damage", 0))
        rof = float(getattr(tower_def, "rof", 0))
        max_cost = max(max_cost, base_cost + max_upgrades * (base_cost / 2.0))
        max_range = max(max_range, base_range + max_upgrades * (base_range / 20.0))
        max_damage = max(max_damage, base_damage + max_upgrades * (base_damage / 2.2))
        max_rof = max(max_rof, rof)
    return max_cost, max_range, max_damage, max_rof


def _tower_slot_features(spec: ActionSpaceSpec) -> list[str]:
    features = ["exists", "cell_x_norm", "cell_y_norm"]
    features.extend([f"kind_{kind}" for kind in spec.tower_kinds])
    features.append("level_norm")
    features.extend([f"mode_{mode}" for mode in spec.target_modes])
    features.extend(["cost_norm", "range_norm", "damage_norm", "rof_norm", "cooldown_norm"])
    return features


def build_observation(state, map_data, spec: ActionSpaceSpec) -> dict[str, Any]:
    wave_info = wave_display_info(state, map_data)
    current = wave_info.get("current") or {}
    next_wave = wave_info.get("next") or {}

    kind_to_idx = {kind: idx for idx, kind in enumerate(spec.tower_kinds)}
    mode_to_idx = {mode: idx for idx, mode in enumerate(spec.target_modes)}
    max_cost, max_range, max_damage, max_rof = _max_tower_stats(MAX_LEVEL)
    max_cooldown = max(max_rof, 10.0)
    max_lives = max(1, MAX_LIVES)
    max_waves = max(1, len(LEVELS))
    max_creep_type = max(1, max(LEVELS) if LEVELS else 1)

    width = float(getattr(map_data, "width", 1.0) or 1.0)
    height = float(getattr(map_data, "height", 1.0) or 1.0)

    bank = int(getattr(state, "bank", 0))
    score = int(getattr(state, "score", 0))
    bank_scale = max_cost * float(spec.max_towers)
    score_scale = max(bank_scale * SCORE_SCALE_MULTIPLIER, 1.0)

    towers = sorted_towers(state)
    tower_slots: list[list[float]] = []
    slot_features = _tower_slot_features(spec)
    empty_slot = [0.0] * len(slot_features)
    for slot_idx in range(spec.max_towers):
        if slot_idx < len(towers):
            tower = towers[slot_idx]
            kind_idx = kind_to_idx.get(str(getattr(tower, "kind", "")), -1)
            mode_idx = mode_to_idx.get(str(getattr(tower, "target_mode", "")), -1)
            kind_onehot = [0.0] * len(spec.tower_kinds)
            if kind_idx >= 0:
                kind_onehot[kind_idx] = 1.0
            mode_onehot = [0.0] * len(spec.target_modes)
            if mode_idx >= 0:
                mode_onehot[mode_idx] = 1.0
            tower_slots.append(
                [
                    1.0,
                    min(1.0, float(getattr(tower, "cell_x", 0)) / width),
                    min(1.0, float(getattr(tower, "cell_y", 0)) / height),
                    *kind_onehot,
                    min(1.0, float(getattr(tower, "level", 0)) / MAX_LEVEL),
                    *mode_onehot,
                    _log_norm(getattr(tower, "cost", 0), max_cost),
                    min(1.0, float(getattr(tower, "range", 0)) / max_range),
                    _log_norm(getattr(tower, "damage", 0), max_damage),
                    min(1.0, float(getattr(tower, "rof", 0)) / max_rof),
                    min(1.0, float(getattr(tower, "cooldown", 0.0)) / max_cooldown),
                ]
            )
        else:
            tower_slots.append(list(empty_slot))

    current_type = int(current.get("type", -1) or -1)
    next_type = int(next_wave.get("type", -1) or -1)
    current_hp = int(current.get("hp", 0) or 0)
    next_hp = int(next_wave.get("hp", 0) or 0)

    obs: dict[str, Any] = {
        "bank": bank,
        "lives": int(getattr(state, "lives", 0)),
        "score": score,
        "wave": int(getattr(state, "level", 0)),
        "interest": int(getattr(state, "interest", 0)),
        "base_hp": int(getattr(state, "base_hp", 0)),
        "base_worth": int(getattr(state, "base_worth", 0)),
        "wave_current_type": current_type,
        "wave_current_hp": current_hp,
        "wave_next_type": next_type,
        "wave_next_hp": next_hp,
        "tower_count": int(len(getattr(state, "towers", []) or [])),
        "bank_norm": _log_norm(bank, bank_scale),
        "lives_norm": min(1.0, float(getattr(state, "lives", 0)) / max_lives),
        "score_norm": _log_norm(score, score_scale),
        "wave_norm": min(1.0, float(getattr(state, "level", 0)) / max_waves),
        "interest_norm": min(1.0, float(getattr(state, "interest", 0)) / INTEREST_SCALE),
        "base_hp_norm": _log_norm(getattr(state, "base_hp", 0), HP_SCALE),
        "base_worth_norm": min(1.0, float(getattr(state, "base_worth", 0)) / BASE_WORTH_SCALE),
        "tower_count_norm": min(1.0, float(len(getattr(state, "towers", []) or [])) / spec.max_towers),
        "wave_current_present": 1.0 if current_type >= 0 else 0.0,
        "wave_next_present": 1.0 if next_type >= 0 else 0.0,
        "wave_current_type_norm": min(1.0, float(max(0, current_type)) / max_creep_type),
        "wave_next_type_norm": min(1.0, float(max(0, next_type)) / max_creep_type),
        "wave_current_hp_norm": _log_norm(current_hp, HP_SCALE),
        "wave_next_hp_norm": _log_norm(next_hp, HP_SCALE),
        "tower_slots": tower_slots,
        "tower_slot_features": slot_features,
    }
    return obs
