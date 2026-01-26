from __future__ import annotations

from typing import Any
import math

from vectortd.core.model.towers import is_buff_tower, list_tower_defs
from vectortd.core.rules.placement import cell_is_buildable
from vectortd.core.rules.wave_spawner import LEVELS, wave_display_info

from .actions import ActionSpaceSpec, sorted_towers


MAX_LEVEL = 10
MAX_LIVES = 20
INTEREST_SCALE = 10.0
BASE_WORTH_SCALE = 50.0
HP_SCALE = 1_000_000.0
SCORE_SCALE_MULTIPLIER = 10.0
PLACE_CANDIDATE_FEATURES = (
    "valid",
    "affordable",
    "cost_norm",
    "range_norm",
    "coverage",
    "dist_to_path_norm",
    "near_spawn_norm",
    "near_end_norm",
    "local_density",
    "tower_count_in_buff_radius",
    "is_buffD",
    "is_buffR",
)


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


def _cell_within_bounds(map_data, cell_x: int, cell_y: int) -> bool:
    grid = int(getattr(map_data, "grid", 25))
    width = int(getattr(map_data, "width", 0))
    height = int(getattr(map_data, "height", 0))
    if grid <= 0 or width <= 0 or height <= 0:
        return False
    if cell_x < 0 or cell_y < 0:
        return False
    return cell_x + grid <= width and cell_y + grid <= height


def build_observation(
    state,
    map_data,
    spec: ActionSpaceSpec,
    *,
    place_candidate_cells: list[tuple[int, int] | None] | None = None,
    place_candidate_centers: list[tuple[float, float] | None] | None = None,
    place_candidate_static: list[tuple[float, ...]] | None = None,
    place_candidate_tower_idx: list[int] | None = None,
) -> dict[str, Any]:
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
    ups_value = int(getattr(state, "ups", 0))
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
        "ups": ups_value,
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
        "ups_norm": float(ups_value) / (float(ups_value) + 1.0),
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

    if (
        place_candidate_cells is not None
        and place_candidate_static is not None
        and place_candidate_tower_idx is not None
    ):
        towers = list(getattr(state, "towers", []) or [])
        can_build = len(towers) < int(getattr(spec, "max_towers", len(towers) + 1))
        occupied = {(int(getattr(tower, "cell_x", 0)), int(getattr(tower, "cell_y", 0))) for tower in towers}
        grid = int(getattr(map_data, "grid", 25))
        tower_centers = [
            (float(getattr(tower, "cell_x", 0)) + grid * 0.5, float(getattr(tower, "cell_y", 0)) + grid * 0.5)
            for tower in towers
        ]
        buff_radius_sq = 100.0 * 100.0
        max_towers_norm = max(1, int(getattr(spec, "max_towers", 1)))
        ups = int(getattr(state, "ups", 0))

        if place_candidate_centers is None:
            place_candidate_centers = []
            for cell in place_candidate_cells:
                if cell is None:
                    place_candidate_centers.append(None)
                    continue
                cell_x, cell_y = cell
                place_candidate_centers.append((cell_x + grid * 0.5, cell_y + grid * 0.5))

        place_candidates: list[list[float]] = []
        for idx, cell in enumerate(place_candidate_cells):
            if cell is None or idx >= len(place_candidate_static):
                place_candidates.append([0.0] * len(PLACE_CANDIDATE_FEATURES))
                continue
            static = place_candidate_static[idx]
            tower_idx = place_candidate_tower_idx[idx] if idx < len(place_candidate_tower_idx) else -1
            kind = (
                str(spec.tower_kinds[tower_idx])
                if 0 <= tower_idx < len(spec.tower_kinds)
                else ""
            )
            is_buff = is_buff_tower(kind)
            cost = int(spec.tower_costs[tower_idx]) if 0 <= tower_idx < len(spec.tower_costs) else 0

            cell_x, cell_y = cell
            valid = (
                can_build
                and _cell_within_bounds(map_data, cell_x, cell_y)
                and cell_is_buildable(map_data, cell_x, cell_y)
                and (cell_x, cell_y) not in occupied
            )
            if is_buff:
                affordable = ups >= 1
            else:
                affordable = bank >= cost

            center = place_candidate_centers[idx] if idx < len(place_candidate_centers) else None
            if center is None:
                tower_count_norm = 0.0
            else:
                cx, cy = center
                count = 0
                for tx, ty in tower_centers:
                    dx = tx - cx
                    dy = ty - cy
                    if dx * dx + dy * dy <= buff_radius_sq:
                        count += 1
                tower_count_norm = min(1.0, float(count) / float(max_towers_norm))

            cost_norm = static[0] if len(static) > 0 else 0.0
            range_norm = static[1] if len(static) > 1 else 0.0
            coverage = static[2] if len(static) > 2 else 0.0
            dist_to_path_norm = static[3] if len(static) > 3 else 0.0
            near_spawn_norm = static[4] if len(static) > 4 else 0.0
            near_end_norm = static[5] if len(static) > 5 else 0.0
            local_density = static[6] if len(static) > 6 else 0.0
            is_buff_d = static[7] if len(static) > 7 else 0.0
            is_buff_r = static[8] if len(static) > 8 else 0.0

            place_candidates.append(
                [
                    1.0 if valid else 0.0,
                    1.0 if affordable else 0.0,
                    float(cost_norm),
                    float(range_norm),
                    float(coverage),
                    float(dist_to_path_norm),
                    float(near_spawn_norm),
                    float(near_end_norm),
                    float(local_density),
                    float(tower_count_norm),
                    float(is_buff_d),
                    float(is_buff_r),
                ]
            )

        obs["place_candidates"] = place_candidates
        obs["place_candidate_features"] = list(PLACE_CANDIDATE_FEATURES)
    return obs
