from __future__ import annotations

from numbers import Integral
from pathlib import Path
import logging
import math
import time
from typing import Any

import gymnasium as gym
import numpy as np

from vectortd.core.engine import Engine
from vectortd.core.model.map import load_map_json
from vectortd.core.model.towers import BUFF_TOWER_KINDS, get_tower_def, list_tower_defs
from vectortd.core.rules.placement import buildable_cells
from vectortd.core.rng import seed_state

from .action_space.candidates import compute_path_samples, score_cells_for_type, select_top_k_diverse
from .action_space.discrete_k import (
    DiscreteKActionTable,
    DiscreteKSpec,
    OP_NOOP,
    OP_PLACE,
    OP_SELL,
    OP_SET_MODE,
    OP_START_WAVE,
    OP_UPGRADE,
    build_discrete_k_spec,
)
from .actions import (
    Action,
    MAX_CELLS,
    MAX_TOWERS,
    Noop,
    Place,
    Sell,
    SetMode,
    StartWave,
    Upgrade,
    action_space_spec,
    flatten,
    get_tower_slots,
    unflatten,
)
from .masking import compute_action_mask, compute_action_mask_discrete_k
from .obs import PLACE_CANDIDATE_FEATURES, _tower_slot_features, build_observation
from .obs_flatten import SCALAR_KEYS, flatten_observation
from .rewards import RewardConfig, RewardState, compute_reward_breakdown, reward_state_from


logger = logging.getLogger(__name__)


DISCRETE_K_TARGET_MODES = ("closest", "weakest", "hardest", "fastest", "random")


def _kcells_for_range(range_px: int) -> int:
    if range_px >= 150:
        return 64
    if range_px >= 100:
        return 48
    return 32


def _dmin_cells_for_range(range_px: int) -> int:
    return 2 if range_px >= 150 else 1


def _effective_range_px(kind: str, base_range: int) -> float:
    if str(kind) in BUFF_TOWER_KINDS:
        return 100.0
    return float(base_range)


def _log_norm(value: int | float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0.0, float(value))) / math.log1p(scale))


def _discrete_k_tower_defs():
    tower_defs = list(list_tower_defs())
    for kind in BUFF_TOWER_KINDS:
        tower_defs.append(get_tower_def(kind))
    return tower_defs


def _resolve_map_path(map_path: str) -> Path:
    p = Path(map_path)
    if p.suffix:
        return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p
    if p.parent == Path("."):
        return Path(__file__).resolve().parents[3] / "data/maps" / f"{p.name}.json"
    return Path(__file__).resolve().parents[3] / p.with_suffix(".json")


class VectorTDEventEnv(gym.Env):
    def __init__(
        self,
        *,
        default_map: str = "switchback",
        action_space_kind: str = "legacy",
        max_towers: int = MAX_TOWERS,
        max_cells: int = MAX_CELLS,
        max_wave_ticks: int = 20000,
        max_build_actions: int | None = 100,
        place_cell_top_k: int | None = None,
        strict_invalid_actions: bool = False,
        reward_config: RewardConfig | None = None,
        include_action_mask_in_obs: bool = False,
        timing_enabled: bool = False,
        log_dir: str | Path | None = None,
        log_prefix: str | None = None,
        log_interval_sec: float = 10.0,
        log_every_wave: bool = True,
        log_every_reset: bool = True,
        debug_actions: bool = False,
    ) -> None:
        super().__init__()
        self.default_map = default_map
        self.action_space_kind = str(action_space_kind)
        if self.action_space_kind not in ("legacy", "discrete_k"):
            raise ValueError(f"Unsupported action_space_kind={self.action_space_kind!r}")
        self.max_towers = max_towers
        self.max_cells = max_cells
        self.max_wave_ticks = max_wave_ticks
        self.max_build_actions = max_build_actions
        self.place_cell_top_k = None if place_cell_top_k is None else int(place_cell_top_k)
        self.strict_invalid_actions = strict_invalid_actions
        self.include_action_mask_in_obs = include_action_mask_in_obs
        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_config = reward_config

        self.timing_enabled = timing_enabled
        self.timing: dict[str, float] = {}

        self.engine: Engine | None = None
        self.map_data = None
        self.action_spec = None
        self._discrete_k_spec: DiscreteKSpec | None = None
        self._action_table = None
        self._action_obj_table: list[Action] | None = None
        self._action_id_by_action: dict[Action, int] | None = None
        self._place_candidate_cells: list[tuple[int, int] | None] | None = None
        self._place_candidate_centers: list[tuple[float, float] | None] | None = None
        self._place_candidate_static: list[tuple[float, ...]] | None = None
        self._place_candidate_tower_idx: list[int] | None = None
        self.phase = "BUILD"

        self._prev_lives = 0
        self._prev_wave = 0
        self._episode_return = 0.0

        self.episode_actions: list[list[Action]] = []
        self._current_wave_actions: list[Action] = []
        self.episode_seed: int | None = None
        self.map_id: str | None = None
        self.map_path: str | None = None
        self.build_actions_since_wave = 0
        self._last_action_mask: np.ndarray | None = None
        self._last_obs_dict: dict[str, Any] | None = None
        self._obs_dim = 1
        self._initial_seed: int | None = None
        self._initial_seed_used = False
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self._bootstrap_spaces()

        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.log_prefix = log_prefix or "env"
        self.log_interval_sec = float(log_interval_sec)
        self.log_every_wave = bool(log_every_wave)
        self.log_every_reset = bool(log_every_reset)
        self._log_handle = None
        self._last_log_time = 0.0
        self._step_count = 0
        self._last_action_id: int | None = None
        self.debug_actions = bool(debug_actions)
        self._debug_action_logs = 0
        self.engine_seed: int | None = None
        self.reset_seed: int | None = None
        self._debug_non_finite_obs_logged = False
        self._debug_non_finite_mask_logged = False
        self._debug_empty_mask_logged = False
        self._debug_forced_start_wave_logged = False
        self._debug_wave_action_stats: dict[str, Any] | None = None
        self._debug_episode_action_stats: dict[str, Any] | None = None
        self._debug_wave_mask_stats: dict[str, Any] | None = None
        self._debug_episode_mask_stats: dict[str, Any] | None = None
        self._debug_negative_bank_logged = False
        self._debug_negative_lives_logged = False
        self._debug_mask_violation_logged = False

    def _debug_context(self) -> str:
        state = self.engine.state if self.engine is not None else None
        wave = int(getattr(state, "level", 0)) if state is not None else 0
        lives = int(getattr(state, "lives", 0)) if state is not None else 0
        score = int(getattr(state, "score", 0)) if state is not None else 0
        bank = int(getattr(state, "bank", 0)) if state is not None else 0
        rng_calls = int(getattr(state, "rng_calls", 0)) if state is not None else 0
        return (
            "map_id={} map_path={} reset_seed={} engine_seed={} episode_seed={} step={} phase={} "
            "wave={} lives={} score={} bank={} rng_calls={} action_id={}"
        ).format(
            self.map_id,
            self.map_path,
            self.reset_seed,
            self.engine_seed,
            self.episode_seed,
            self._step_count,
            self.phase,
            wave,
            lives,
            score,
            bank,
            rng_calls,
            self._last_action_id,
        )

    def _using_discrete_k(self) -> bool:
        return self.action_space_kind == "discrete_k"

    def _action_from_discrete_k_spec(self, spec) -> Action:
        if spec.op == OP_NOOP:
            return Noop()
        if spec.op == OP_START_WAVE:
            return StartWave()
        if spec.op == OP_PLACE:
            if spec.t is None or spec.k is None:
                return Noop()
            return Place(tower_type=int(spec.t), cell=int(spec.k))
        if spec.op == OP_UPGRADE:
            if spec.slot is None:
                return Noop()
            return Upgrade(tower_id=int(spec.slot))
        if spec.op == OP_SELL:
            if spec.slot is None:
                return Noop()
            return Sell(tower_id=int(spec.slot))
        if spec.op == OP_SET_MODE:
            if spec.slot is None or spec.mode is None:
                return Noop()
            return SetMode(tower_id=int(spec.slot), mode=int(spec.mode))
        return Noop()

    def _build_discrete_k_data(self, map_data):
        tower_defs = _discrete_k_tower_defs()
        tower_kinds = tuple(str(tower_def.kind) for tower_def in tower_defs)
        tower_costs = tuple(int(getattr(tower_def, "cost", 0)) for tower_def in tower_defs)
        target_modes = tuple(DISCRETE_K_TARGET_MODES)
        kind_to_mode_mask = {
            str(tower_def.kind): tuple(mode in tower_def.target_modes for mode in target_modes)
            for tower_def in tower_defs
        }
        kcells_by_type = tuple(_kcells_for_range(int(getattr(tower_def, "range", 0))) for tower_def in tower_defs)

        buildable = list(buildable_cells(map_data))
        path_samples = compute_path_samples(map_data, step_px=10.0)
        grid = int(getattr(map_data, "grid", 25))
        centers = [(cell[0] + grid * 0.5, cell[1] + grid * 0.5) for cell in buildable]
        cell_center = {cell: center for cell, center in zip(buildable, centers)}
        total_cells = len(buildable)
        local_density: dict[tuple[int, int], float] = {}
        if total_cells:
            radius_sq = float(2 * grid) ** 2
            for cell, center in zip(buildable, centers):
                count = 0
                cx, cy = center
                for ox, oy in centers:
                    dx = ox - cx
                    dy = oy - cy
                    if dx * dx + dy * dy <= radius_sq:
                        count += 1
                local_density[cell] = float(count) / float(total_cells)

        spawn_point = path_samples[0] if path_samples else None
        end_point = path_samples[-1] if path_samples else None
        near_spawn: dict[tuple[int, int], float] = {}
        near_end: dict[tuple[int, int], float] = {}
        if spawn_point and end_point:
            sx, sy = spawn_point
            ex, ey = end_point
            for cell, center in zip(buildable, centers):
                cx, cy = center
                spawn_dist = math.hypot(cx - sx, cy - sy)
                end_dist = math.hypot(cx - ex, cy - ey)
                near_spawn[cell] = min(1.0, spawn_dist / 600.0)
                near_end[cell] = min(1.0, end_dist / 600.0)

        cells_by_type: dict[int, list[tuple[int, int] | None]] = {}
        candidate_cells_flat: list[tuple[int, int] | None] = []
        candidate_centers_flat: list[tuple[float, float] | None] = []
        candidate_static_flat: list[tuple[float, ...]] = []
        candidate_tower_idx_flat: list[int] = []
        for idx, tower_def in enumerate(tower_defs):
            base_range = int(getattr(tower_def, "range", 0))
            kcells = kcells_by_type[idx]
            effective_range = _effective_range_px(tower_def.kind, base_range)
            scored = score_cells_for_type(
                buildable,
                path_samples,
                effective_range,
                grid=grid,
            )
            scored_by_cell = {entry.cell: entry for entry in scored}
            dmin_cells = _dmin_cells_for_range(base_range)
            selected = select_top_k_diverse(scored, k=kcells, dmin_px=dmin_cells * grid)
            if len(selected) < kcells:
                selected.extend([None] * (kcells - len(selected)))
            cells_by_type[idx] = selected
            cost_norm = _log_norm(getattr(tower_def, "cost", 0), 5000.0)
            range_norm = min(1.0, float(base_range) / 200.0)
            is_buff_d = 1.0 if str(tower_def.kind) == "buffD" else 0.0
            is_buff_r = 1.0 if str(tower_def.kind) == "buffR" else 0.0
            for cell in selected:
                candidate_tower_idx_flat.append(idx)
                candidate_cells_flat.append(cell)
                if cell is None:
                    candidate_centers_flat.append(None)
                    candidate_static_flat.append((0.0,) * 9)
                    continue
                entry = scored_by_cell.get(cell)
                coverage = float(entry.coverage) if entry is not None else 0.0
                dist_to_path = float(entry.dist_to_path) if entry is not None else 0.0
                dist_to_path_norm = min(1.0, dist_to_path / 200.0)
                near_spawn_norm = near_spawn.get(cell, 0.0)
                near_end_norm = near_end.get(cell, 0.0)
                density = local_density.get(cell, 0.0)
                candidate_centers_flat.append(cell_center.get(cell))
                candidate_static_flat.append(
                    (
                        float(cost_norm),
                        float(range_norm),
                        float(coverage),
                        float(dist_to_path_norm),
                        float(near_spawn_norm),
                        float(near_end_norm),
                        float(density),
                        float(is_buff_d),
                        float(is_buff_r),
                    )
                )

        table_builder = DiscreteKActionTable(
            tower_types=tower_kinds,
            kcells_by_type=kcells_by_type,
            ktower=self.max_towers,
            modes=target_modes,
        )
        action_space, action_table, normalized_cells = table_builder.build_for_map(
            map_data,
            cells_by_type=cells_by_type,
        )
        cells_by_type_list = [normalized_cells.get(idx, []) for idx in range(len(tower_kinds))]
        spec = build_discrete_k_spec(
            max_towers=self.max_towers,
            map_name=str(getattr(map_data, "name", "")),
            tower_kinds=tower_kinds,
            tower_costs=tower_costs,
            target_modes=target_modes,
            kind_to_mode_mask=kind_to_mode_mask,
            kcells_by_type=kcells_by_type,
            cells_by_type=cells_by_type_list,
        )
        action_obj_table = [self._action_from_discrete_k_spec(item) for item in action_table]
        action_id_by_action = {action: idx for idx, action in enumerate(action_obj_table)}
        return (
            spec,
            action_space,
            action_table,
            action_obj_table,
            action_id_by_action,
            candidate_cells_flat,
            candidate_centers_flat,
            candidate_static_flat,
            candidate_tower_idx_flat,
        )

    def _debug_check_finite(self, *, kind: str, array: Any) -> None:
        if not self.debug_actions:
            return
        flag_attr = "_debug_non_finite_obs_logged" if kind == "obs" else "_debug_non_finite_mask_logged"
        if getattr(self, flag_attr, False):
            return
        try:
            data = np.asarray(array, dtype=float)
        except (TypeError, ValueError):
            data = np.asarray(array)
        if data.size == 0:
            return
        try:
            finite = np.isfinite(data)
        except TypeError:
            return
        if finite.all():
            return
        non_finite_count = int((~finite).sum())
        first_idx = None
        first_val = None
        try:
            first_idx = np.argwhere(~finite)[0].tolist()
            first_val = data[tuple(first_idx)] if data.ndim > 0 else data.item()
        except Exception:
            pass
        finite_vals = data[finite]
        finite_min = float(finite_vals.min()) if finite_vals.size else None
        finite_max = float(finite_vals.max()) if finite_vals.size else None
        self._log_line(
            "debug_non_finite kind={} count={} shape={} first_index={} first_value={} finite_min={} finite_max={} {}".format(
                kind,
                non_finite_count,
                tuple(data.shape),
                first_idx,
                first_val,
                finite_min,
                finite_max,
                self._debug_context(),
            )
        )
        setattr(self, flag_attr, True)

    def _debug_new_action_stats(self) -> dict[str, Any]:
        return {
            "total": 0,
            "invalid": 0,
            "forced_start_wave": 0,
            "build_action_limit_violation": 0,
            "noop": 0,
            "start_wave": 0,
            "place": 0,
            "upgrade": 0,
            "sell": 0,
            "set_mode": 0,
            "place_by_kind": {},
            "place_by_cell": {},
            "set_mode_by_mode": {},
        }

    def _debug_new_mask_stats(self) -> dict[str, Any]:
        return {
            "steps": 0,
            "valid_total_sum": 0,
            "valid_total_min": None,
            "valid_total_max": None,
            "noop_only_steps": 0,
            "empty_steps": 0,
            "start_wave_allowed_steps": 0,
            "place_sum": 0,
            "upgrade_sum": 0,
            "sell_sum": 0,
            "set_mode_sum": 0,
        }

    def _debug_reset_episode_stats(self) -> None:
        if not self.debug_actions:
            return
        self._debug_episode_action_stats = self._debug_new_action_stats()
        self._debug_episode_mask_stats = self._debug_new_mask_stats()
        self._debug_reset_wave_stats()
        self._debug_negative_bank_logged = False
        self._debug_negative_lives_logged = False
        self._debug_mask_violation_logged = False

    def _debug_reset_wave_stats(self) -> None:
        if not self.debug_actions:
            return
        self._debug_wave_action_stats = self._debug_new_action_stats()
        self._debug_wave_mask_stats = self._debug_new_mask_stats()

    def _debug_mask_counts(self, mask: Any) -> dict[str, int]:
        spec = self.action_spec
        if spec is None:
            return {
                "valid_total": 0,
                "noop": 0,
                "start_wave": 0,
                "place": 0,
                "upgrade": 0,
                "sell": 0,
                "set_mode": 0,
            }
        mask_array = np.asarray(mask, dtype=bool)

        def _slice_sum(start: int, count: int) -> int:
            if count <= 0:
                return 0
            end = min(start + count, mask_array.shape[0])
            if start >= end:
                return 0
            return int(mask_array[start:end].sum())

        noop = int(mask_array[spec.offsets.noop]) if spec.offsets.noop < mask_array.shape[0] else 0
        start_wave = (
            int(mask_array[spec.offsets.start_wave]) if spec.offsets.start_wave < mask_array.shape[0] else 0
        )
        place = _slice_sum(spec.offsets.place, spec.place_count)
        upgrade = _slice_sum(spec.offsets.upgrade, spec.upgrade_count)
        sell = _slice_sum(spec.offsets.sell, spec.sell_count)
        set_mode = _slice_sum(spec.offsets.set_mode, spec.set_mode_count)
        return {
            "valid_total": int(mask_array.sum()),
            "noop": noop,
            "start_wave": start_wave,
            "place": place,
            "upgrade": upgrade,
            "sell": sell,
            "set_mode": set_mode,
        }

    def _debug_update_mask_stats(self, mask: Any) -> None:
        if not self.debug_actions:
            return
        counts = self._debug_mask_counts(mask)
        for stats in (self._debug_wave_mask_stats, self._debug_episode_mask_stats):
            if stats is None:
                continue
            stats["steps"] += 1
            stats["valid_total_sum"] += counts["valid_total"]
            if stats["valid_total_min"] is None or counts["valid_total"] < stats["valid_total_min"]:
                stats["valid_total_min"] = counts["valid_total"]
            if stats["valid_total_max"] is None or counts["valid_total"] > stats["valid_total_max"]:
                stats["valid_total_max"] = counts["valid_total"]
            if counts["valid_total"] == 0:
                stats["empty_steps"] += 1
            if counts["valid_total"] == 1 and counts["noop"] == 1:
                stats["noop_only_steps"] += 1
            stats["start_wave_allowed_steps"] += counts["start_wave"]
            stats["place_sum"] += counts["place"]
            stats["upgrade_sum"] += counts["upgrade"]
            stats["sell_sum"] += counts["sell"]
            stats["set_mode_sum"] += counts["set_mode"]

    def _debug_record_action(
        self,
        action_obj: Action,
        *,
        invalid_action: bool,
        build_action_limit_violation: bool,
        forced_start_wave: bool,
    ) -> None:
        if not self.debug_actions:
            return
        for stats in (self._debug_wave_action_stats, self._debug_episode_action_stats):
            if stats is None:
                continue
            stats["total"] += 1
            if invalid_action:
                stats["invalid"] += 1
            if forced_start_wave:
                stats["forced_start_wave"] += 1
            if build_action_limit_violation:
                stats["build_action_limit_violation"] += 1
            if isinstance(action_obj, Noop):
                stats["noop"] += 1
            elif isinstance(action_obj, StartWave):
                stats["start_wave"] += 1
            elif isinstance(action_obj, Place):
                stats["place"] += 1
                spec = self.action_spec
                if spec is not None and 0 <= action_obj.tower_type < len(spec.tower_kinds):
                    kind = str(spec.tower_kinds[action_obj.tower_type])
                    stats["place_by_kind"][kind] = stats["place_by_kind"].get(kind, 0) + 1
                stats["place_by_cell"][int(action_obj.cell)] = stats["place_by_cell"].get(int(action_obj.cell), 0) + 1
            elif isinstance(action_obj, Upgrade):
                stats["upgrade"] += 1
            elif isinstance(action_obj, Sell):
                stats["sell"] += 1
            elif isinstance(action_obj, SetMode):
                stats["set_mode"] += 1
                spec = self.action_spec
                if spec is not None and 0 <= action_obj.mode < len(spec.target_modes):
                    mode = str(spec.target_modes[action_obj.mode])
                    stats["set_mode_by_mode"][mode] = stats["set_mode_by_mode"].get(mode, 0) + 1

    def _debug_check_ranges(self, state: RewardState) -> None:
        if not self.debug_actions:
            return
        if state.bank < 0 and not self._debug_negative_bank_logged:
            self._log_line(f"debug_invariant bank_negative bank={state.bank} {self._debug_context()}")
            self._debug_negative_bank_logged = True
        if state.lives < 0 and not self._debug_negative_lives_logged:
            self._log_line(f"debug_invariant lives_negative lives={state.lives} {self._debug_context()}")
            self._debug_negative_lives_logged = True

    def _debug_validate_mask(self, mask: Any, *, state: Any, phase: str) -> None:
        if not self.debug_actions or self._debug_mask_violation_logged:
            return
        spec = self.action_spec
        if spec is None:
            return
        if not hasattr(spec, "cell_positions"):
            return
        mask_array = np.asarray(mask, dtype=bool)
        build_limit_reached = self._build_action_limit_reached()
        start_idx = spec.offsets.start_wave
        start_wave_allowed = start_idx < mask_array.shape[0] and bool(mask_array[start_idx])
        if mask_array.shape[0] != spec.num_actions:
            self._log_line(
                "debug_mask_invalid kind=length expected={} actual={} {}".format(
                    spec.num_actions,
                    mask_array.shape[0],
                    self._debug_context(),
                )
            )
            self._debug_mask_violation_logged = True
            return
        noop_idx = spec.offsets.noop
        if noop_idx >= mask_array.shape[0] or not bool(mask_array[noop_idx]):
            if build_limit_reached and start_wave_allowed:
                pass
            else:
                self._log_line("debug_mask_invalid kind=noop_missing {}".format(self._debug_context()))
                self._debug_mask_violation_logged = True
                return

        if phase != "BUILD" or bool(getattr(state, "game_over", False)):
            non_noop = mask_array.copy()
            non_noop[noop_idx] = False
            if bool(non_noop.any()):
                self._log_line("debug_mask_invalid kind=non_build_actions {}".format(self._debug_context()))
                self._debug_mask_violation_logged = True
            return

        towers = list(getattr(state, "towers", []) or [])
        can_build = len(towers) < spec.max_towers
        place_base = spec.offsets.place
        place_end = place_base + spec.place_count
        if place_base < mask_array.shape[0]:
            place_slice = mask_array[place_base:min(place_end, mask_array.shape[0])]
        else:
            place_slice = np.zeros(0, dtype=bool)
        if not can_build and bool(place_slice.any()):
            self._log_line("debug_mask_invalid kind=place_no_capacity {}".format(self._debug_context()))
            self._debug_mask_violation_logged = True
            return
        bank_value = getattr(state, "bank", None)
        bank_value = int(bank_value) if bank_value is not None else None
        cell_count = len(spec.cell_positions)
        if can_build and cell_count > 0 and place_slice.size:
            occupied = np.zeros(cell_count, dtype=bool)
            cell_index = spec.cell_index
            for tower in towers:
                cell_x = int(getattr(tower, "cell_x", -1))
                cell_y = int(getattr(tower, "cell_y", -1))
                idx = cell_index.get((cell_x, cell_y))
                if idx is not None:
                    occupied[idx] = True
            for tower_type, cost in enumerate(spec.tower_costs):
                base = place_base + tower_type * cell_count
                end = base + cell_count
                if base >= mask_array.shape[0]:
                    break
                slice_mask = mask_array[base:min(end, mask_array.shape[0])]
                if bank_value is None or bank_value < cost:
                    if bool(slice_mask.any()):
                        self._log_line(
                            "debug_mask_invalid kind=place_bank tower_type={} cost={} bank={} {}".format(
                                tower_type,
                                cost,
                                bank_value,
                                self._debug_context(),
                            )
                        )
                        self._debug_mask_violation_logged = True
                        return
                else:
                    bad = slice_mask & occupied[: slice_mask.shape[0]]
                    if bool(bad.any()):
                        self._log_line(
                            "debug_mask_invalid kind=place_occupied tower_type={} {}".format(
                                tower_type,
                                self._debug_context(),
                            )
                        )
                        self._debug_mask_violation_logged = True
                        return

        tower_slots = get_tower_slots(state, spec.max_towers)
        mode_count = len(spec.target_modes)
        for slot_idx, tower in enumerate(tower_slots):
            upgrade_idx = spec.offsets.upgrade + slot_idx
            sell_idx = spec.offsets.sell + slot_idx
            if tower is None:
                if upgrade_idx < mask_array.shape[0] and bool(mask_array[upgrade_idx]):
                    self._log_line("debug_mask_invalid kind=upgrade_empty_slot {}".format(self._debug_context()))
                    self._debug_mask_violation_logged = True
                    return
                if sell_idx < mask_array.shape[0] and bool(mask_array[sell_idx]):
                    self._log_line("debug_mask_invalid kind=sell_empty_slot {}".format(self._debug_context()))
                    self._debug_mask_violation_logged = True
                    return
                if mode_count:
                    base = spec.offsets.set_mode + slot_idx * mode_count
                    if base < mask_array.shape[0] and bool(mask_array[base : min(base + mode_count, mask_array.shape[0])].any()):
                        self._log_line("debug_mask_invalid kind=mode_empty_slot {}".format(self._debug_context()))
                        self._debug_mask_violation_logged = True
                        return
                continue
            if upgrade_idx < mask_array.shape[0]:
                level = int(getattr(tower, "level", 0))
                upgrade_cost = int(getattr(tower, "base_cost", 0) / 2)
                if level >= 10 or bank_value is None or bank_value < upgrade_cost:
                    if bool(mask_array[upgrade_idx]):
                        self._log_line("debug_mask_invalid kind=upgrade_bank_or_level {}".format(self._debug_context()))
                        self._debug_mask_violation_logged = True
                        return
            if mode_count:
                mode_mask = spec.kind_to_mode_mask.get(str(getattr(tower, "kind", "")))
                base = spec.offsets.set_mode + slot_idx * mode_count
                if base < mask_array.shape[0]:
                    slice_mask = mask_array[base:min(base + mode_count, mask_array.shape[0])]
                    if mode_mask is None:
                        if bool(slice_mask.any()):
                            self._log_line("debug_mask_invalid kind=mode_unknown {}".format(self._debug_context()))
                            self._debug_mask_violation_logged = True
                            return
                    else:
                        for mode_idx, supported in enumerate(mode_mask):
                            idx = base + mode_idx
                            if idx >= mask_array.shape[0]:
                                break
                            if bool(mask_array[idx]) and not supported:
                                self._log_line("debug_mask_invalid kind=mode_unsupported {}".format(self._debug_context()))
                                self._debug_mask_violation_logged = True
                                return

    def _debug_format_counts(self, mapping: dict[Any, int], *, limit: int = 5) -> str:
        items = [(key, count) for key, count in mapping.items() if count]
        if not items:
            return "none"
        items.sort(key=lambda item: (-item[1], str(item[0])))
        parts = [f"{key}:{count}" for key, count in items[:limit]]
        return ",".join(parts)

    def _debug_format_place_cells(self, mapping: dict[int, int], *, limit: int = 5) -> str:
        if not mapping:
            return "none"
        spec = self.action_spec
        items = [(cell, count) for cell, count in mapping.items() if count]
        items.sort(key=lambda item: (-item[1], item[0]))
        parts = []
        for cell_idx, count in items[:limit]:
            if spec is not None and hasattr(spec, "cell_positions") and 0 <= cell_idx < len(spec.cell_positions):
                cell_x, cell_y = spec.cell_positions[cell_idx]
                parts.append(f"{cell_x}x{cell_y}:{count}")
            else:
                parts.append(f"{cell_idx}:{count}")
        return ",".join(parts)

    def _debug_log_mask_summary(self, *, prefix: str, wave: int | None, stats: dict[str, Any]) -> None:
        steps = int(stats.get("steps") or 0)
        if steps <= 0:
            self._log_line(f"{prefix} wave={wave} steps=0")
            return
        valid_avg = float(stats["valid_total_sum"]) / float(steps)
        place_avg = float(stats["place_sum"]) / float(steps)
        upgrade_avg = float(stats["upgrade_sum"]) / float(steps)
        sell_avg = float(stats["sell_sum"]) / float(steps)
        set_mode_avg = float(stats["set_mode_sum"]) / float(steps)
        self._log_line(
            "{} wave={} steps={} valid_avg={:.2f} valid_min={} valid_max={} noop_only_steps={} empty_steps={} "
            "start_wave_allowed_steps={} place_avg={:.2f} upgrade_avg={:.2f} sell_avg={:.2f} set_mode_avg={:.2f}".format(
                prefix,
                wave,
                steps,
                valid_avg,
                stats["valid_total_min"],
                stats["valid_total_max"],
                stats["noop_only_steps"],
                stats["empty_steps"],
                stats["start_wave_allowed_steps"],
                place_avg,
                upgrade_avg,
                sell_avg,
                set_mode_avg,
            )
        )

    def _debug_log_action_summary(
        self,
        *,
        prefix: str,
        wave: int | None,
        build_actions: int | None,
        max_build_actions: int | None,
        limit_reached: bool | None,
        stats: dict[str, Any],
    ) -> None:
        counts = "noop:{noop},start_wave:{start_wave},place:{place},upgrade:{upgrade},sell:{sell},set_mode:{set_mode}".format(
            noop=stats["noop"],
            start_wave=stats["start_wave"],
            place=stats["place"],
            upgrade=stats["upgrade"],
            sell=stats["sell"],
            set_mode=stats["set_mode"],
        )
        place_kinds = self._debug_format_counts(stats["place_by_kind"], limit=5)
        place_cells = self._debug_format_place_cells(stats["place_by_cell"], limit=5)
        set_modes = self._debug_format_counts(stats["set_mode_by_mode"], limit=5)
        unique_place_cells = len(stats["place_by_cell"])
        self._log_line(
            "{} wave={} actions={} invalid_actions={} forced_start_wave={} build_action_limit_violation={} "
            "build_actions={} max_build_actions={} limit_reached={} counts={} place_kinds={} place_cells={} "
            "unique_place_cells={} set_modes={}".format(
                prefix,
                wave,
                stats["total"],
                stats["invalid"],
                stats["forced_start_wave"],
                stats["build_action_limit_violation"],
                build_actions,
                max_build_actions,
                limit_reached,
                counts,
                place_kinds,
                place_cells,
                unique_place_cells,
                set_modes,
            )
        )

    def _debug_log_wave_summary(
        self,
        *,
        wave: int,
        build_actions: int,
        max_build_actions: int | None,
        limit_reached: bool,
    ) -> None:
        if self._debug_wave_action_stats is not None:
            self._debug_log_action_summary(
                prefix="debug_wave_actions",
                wave=wave,
                build_actions=build_actions,
                max_build_actions=max_build_actions,
                limit_reached=limit_reached,
                stats=self._debug_wave_action_stats,
            )
        if self._debug_wave_mask_stats is not None:
            self._debug_log_mask_summary(
                prefix="debug_mask_summary",
                wave=wave,
                stats=self._debug_wave_mask_stats,
            )

    def _debug_log_episode_summary(self, *, wave: int) -> None:
        if self._debug_episode_action_stats is not None:
            self._debug_log_action_summary(
                prefix="debug_episode_actions",
                wave=wave,
                build_actions=None,
                max_build_actions=None,
                limit_reached=None,
                stats=self._debug_episode_action_stats,
            )
        if self._debug_episode_mask_stats is not None:
            self._debug_log_mask_summary(
                prefix="debug_episode_mask_summary",
                wave=wave,
                stats=self._debug_episode_mask_stats,
            )

    @property
    def last_obs(self) -> dict[str, Any] | None:
        return self._last_obs_dict

    def _bootstrap_spaces(self) -> None:
        try:
            resolved = _resolve_map_path(self.default_map)
            map_data = load_map_json(resolved)
            if self._using_discrete_k():
                spec, action_space, _, _, _, _, _, _, _ = self._build_discrete_k_data(map_data)
                slot_size = len(_tower_slot_features(spec))
                candidate_dim = len(PLACE_CANDIDATE_FEATURES)
                self._obs_dim = (
                    len(SCALAR_KEYS)
                    + self.max_towers * slot_size
                    + spec.place_count * candidate_dim
                )
                self.action_space = action_space
            else:
                spec = action_space_spec(
                    map_data,
                    max_towers=self.max_towers,
                    max_cells=self.max_cells,
                )
                slot_size = len(_tower_slot_features(spec))
                self._obs_dim = len(SCALAR_KEYS) + self.max_towers * slot_size
                self.action_space = gym.spaces.Discrete(spec.num_actions)
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_dim,),
                dtype=np.float32,
            )
        except Exception as exc:
            logger.warning("Failed to bootstrap spaces for %s: %s", self.default_map, exc)

    def __getstate__(self):
        state = dict(self.__dict__)
        # Avoid pickling open file handles (breaks SubprocVecEnv get_attr).
        state["_log_handle"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_log_handle(self):
        if self.log_dir is None:
            return None
        if self._log_handle is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{self.log_prefix}.log"
            self._log_handle = path.open("a", encoding="utf-8")
        return self._log_handle

    def _log_line(self, message: str) -> None:
        handle = self._ensure_log_handle()
        if handle is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
        self._last_log_time = time.perf_counter()

    def _maybe_log_heartbeat(self) -> None:
        if self.log_interval_sec <= 0:
            return
        now = time.perf_counter()
        if now - self._last_log_time < self.log_interval_sec:
            return
        state = self.engine.state if self.engine is not None else None
        wave = int(getattr(state, "level", 0)) if state is not None else 0
        lives = int(getattr(state, "lives", 0)) if state is not None else 0
        score = int(getattr(state, "score", 0)) if state is not None else 0
        bank = int(getattr(state, "bank", 0)) if state is not None else 0
        valid_actions = None
        start_wave_allowed = None
        if self._last_action_mask is not None and self.action_spec is not None:
            valid_actions = int(np.sum(self._last_action_mask))
            start_idx = self.action_spec.offsets.start_wave
            if 0 <= start_idx < len(self._last_action_mask):
                start_wave_allowed = bool(self._last_action_mask[start_idx])
        self._log_line(
            "heartbeat step={} phase={} wave={} lives={} score={} bank={} action_id={} build_actions={} max_build_actions={} valid_actions={} start_wave_allowed={}".format(
                self._step_count,
                self.phase,
                wave,
                lives,
                score,
                bank,
                self._last_action_id,
                self.build_actions_since_wave,
                self.max_build_actions,
                valid_actions,
                start_wave_allowed,
            )
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        seed_value = seed
        if seed_value is None and self._initial_seed is not None and not self._initial_seed_used:
            seed_value = int(self._initial_seed)
            self._initial_seed_used = True
        elif seed_value is not None:
            self._initial_seed_used = True
        super().reset(seed=seed_value)
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        map_path = self.default_map
        if options:
            map_path = str(options.get("map_path") or map_path)
        resolved = _resolve_map_path(map_path)
        self.map_path = str(resolved)
        self.map_data = load_map_json(resolved)
        self.map_id = str(getattr(self.map_data, "name", ""))

        self.engine = Engine(self.map_data)
        self.engine.timing_enabled = self.timing_enabled
        self.engine.reset()
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.engine_seed = engine_seed
        self.episode_seed = seed_state(self.engine.state, engine_seed)
        self._step_count = 0
        self.reset_seed = seed_value

        if self._using_discrete_k():
            (
                self._discrete_k_spec,
                self.action_space,
                self._action_table,
                self._action_obj_table,
                self._action_id_by_action,
                self._place_candidate_cells,
                self._place_candidate_centers,
                self._place_candidate_static,
                self._place_candidate_tower_idx,
            ) = self._build_discrete_k_data(self.map_data)
            self.action_spec = self._discrete_k_spec
        else:
            self.action_spec = action_space_spec(
                self.map_data,
                max_towers=self.max_towers,
                max_cells=self.max_cells,
            )
            self.action_space = gym.spaces.Discrete(self.action_spec.num_actions)
            self._discrete_k_spec = None
            self._action_table = None
            self._action_obj_table = None
            self._action_id_by_action = None
            self._place_candidate_cells = None
            self._place_candidate_centers = None
            self._place_candidate_static = None
            self._place_candidate_tower_idx = None

        self.phase = "BUILD"
        self._prev_lives = int(getattr(self.engine.state, "lives", 0))
        self._prev_wave = int(getattr(self.engine.state, "level", 0))
        self._episode_return = 0.0

        self.episode_actions = []
        self._current_wave_actions = []
        self.build_actions_since_wave = 0
        self._debug_reset_episode_stats()

        obs_dict = build_observation(
            self.engine.state,
            self.map_data,
            self.action_spec,
            place_candidate_cells=self._place_candidate_cells,
            place_candidate_centers=self._place_candidate_centers,
            place_candidate_static=self._place_candidate_static,
            place_candidate_tower_idx=self._place_candidate_tower_idx,
        )
        obs_dict["phase"] = self.phase
        slot_size = len(obs_dict.get("tower_slot_features", []) or [])
        place_candidates = obs_dict.get("place_candidates", []) or []
        place_candidate_features = obs_dict.get("place_candidate_features", []) or []
        candidate_dim = len(place_candidate_features)
        self._obs_dim = len(SCALAR_KEYS) + self.max_towers * slot_size + len(place_candidates) * candidate_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        obs = np.asarray(
            flatten_observation(obs_dict, max_towers=self.max_towers, slot_size=slot_size),
            dtype=np.float32,
        )
        self._last_obs_dict = obs_dict
        self._last_action_mask = self._compute_action_mask()
        self._debug_check_finite(kind="obs", array=obs)
        self._debug_check_finite(kind="mask", array=self._last_action_mask)
        if self.include_action_mask_in_obs:
            obs_dict["action_mask"] = self._last_action_mask
        if self.timing_enabled:
            self.timing["reset_calls"] = self.timing.get("reset_calls", 0.0) + 1.0
            self.timing["reset_time_total"] = self.timing.get("reset_time_total", 0.0) + (
                time.perf_counter() - start_time
            )
        info = {"engine_seed": engine_seed, "action_mask": self._last_action_mask}
        if self.log_every_reset:
            self._log_line(
                "reset map={} seed={} engine_seed={}".format(
                    self.map_id,
                    seed_value,
                    engine_seed,
                )
            )
        return obs, info

    def _compute_action_mask(self) -> np.ndarray:
        if self.engine is None or self.map_data is None or self.action_spec is None:
            raise RuntimeError("Environment not reset")
        if self._using_discrete_k():
            start_time = time.perf_counter() if self.timing_enabled else 0.0
            mask = compute_action_mask_discrete_k(
                self.engine.state,
                self.engine,
                self.map_data,
                self.action_spec,
                phase=self.phase,
            )
            mask_array = np.asarray(mask, dtype=bool)
            if self.timing_enabled:
                self.timing["mask_calls"] = self.timing.get("mask_calls", 0.0) + 1.0
                self.timing["mask_time_total"] = self.timing.get("mask_time_total", 0.0) + (
                    time.perf_counter() - start_time
                )
            if self._build_action_limit_reached():
                mask_len = len(mask_array)
                forced = np.zeros(mask_len, dtype=bool)
                start_idx = self.action_spec.offsets.start_wave
                if 0 <= start_idx < mask_len and bool(mask_array[start_idx]):
                    forced[start_idx] = True
                    return forced
                if not self._debug_forced_start_wave_logged:
                    self._debug_forced_start_wave_logged = True
                    self._log_line(
                        "mask_force_start_wave_failed step={} phase={} build_actions={} max_build_actions={} start_wave_allowed={}".format(
                            self._step_count,
                            self.phase,
                            self.build_actions_since_wave,
                            self.max_build_actions,
                            bool(mask_array[start_idx]) if 0 <= start_idx < mask_len else False,
                        )
                    )
                return mask_array
            if not np.any(mask_array):
                if not self._debug_empty_mask_logged:
                    self._debug_empty_mask_logged = True
                    self._log_line(
                        "mask_empty step={} phase={} build_actions={} max_build_actions={}".format(
                            self._step_count,
                            self.phase,
                            self.build_actions_since_wave,
                            self.max_build_actions,
                        )
                    )
                noop_idx = self.action_spec.offsets.noop
                if 0 <= noop_idx < len(mask_array):
                    mask_array[noop_idx] = True
            return mask_array
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        mask = compute_action_mask(
            self.engine.state,
            self.engine,
            self.map_data,
            self.action_spec,
            phase=self.phase,
            place_cell_top_k=self.place_cell_top_k,
        )
        mask_array = np.asarray(mask, dtype=bool)
        if self.timing_enabled:
            self.timing["mask_calls"] = self.timing.get("mask_calls", 0.0) + 1.0
            self.timing["mask_time_total"] = self.timing.get("mask_time_total", 0.0) + (
                time.perf_counter() - start_time
            )
        if self._build_action_limit_reached():
            mask_len = len(mask_array)
            forced = np.zeros(mask_len, dtype=bool)
            start_idx = self.action_spec.offsets.start_wave
            if 0 <= start_idx < mask_len and bool(mask_array[start_idx]):
                forced[start_idx] = True
                return forced
            if not self._debug_forced_start_wave_logged:
                self._debug_forced_start_wave_logged = True
                self._log_line(
                    "mask_force_start_wave_failed step={} phase={} build_actions={} max_build_actions={} start_wave_allowed={}".format(
                        self._step_count,
                        self.phase,
                        self.build_actions_since_wave,
                        self.max_build_actions,
                        bool(mask_array[start_idx]) if 0 <= start_idx < mask_len else False,
                    )
                )
            # Fallback: keep original mask to ensure at least one valid action.
            return mask_array
        if not np.any(mask_array):
            if not self._debug_empty_mask_logged:
                self._debug_empty_mask_logged = True
                self._log_line(
                    "mask_empty step={} phase={} build_actions={} max_build_actions={}".format(
                        self._step_count,
                        self.phase,
                        self.build_actions_since_wave,
                        self.max_build_actions,
                    )
                )
            noop_idx = self.action_spec.offsets.noop
            if 0 <= noop_idx < len(mask_array):
                mask_array[noop_idx] = True
        return mask_array

    def action_masks(self) -> np.ndarray:
        if self._last_action_mask is None:
            self._last_action_mask = self._compute_action_mask()
        self._debug_check_finite(kind="mask", array=self._last_action_mask)
        return self._last_action_mask

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def _build_action_limit_reached(self) -> bool:
        if self.max_build_actions is None:
            return False
        return self.build_actions_since_wave >= self.max_build_actions

    def step(
        self, action: Action | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.engine is None or self.map_data is None or self.action_spec is None:
            raise RuntimeError("Environment not reset")
        if self.phase != "BUILD":
            raise RuntimeError(f"step() called in phase={self.phase!r}")
        step_start = time.perf_counter() if self.timing_enabled else 0.0
        self._step_count += 1
        if isinstance(action, np.ndarray):
            if action.shape == ():
                action = action.item()
            elif action.shape == (1,):
                action = action[0].item()
        if isinstance(action, Integral):
            action = int(action)
        if self.debug_actions and self._debug_action_logs < 5:
            self._log_line(
                "debug_action_input type={} dtype={} shape={} value={}".format(
                    type(action).__name__,
                    getattr(action, "dtype", None),
                    getattr(action, "shape", None),
                    action,
                )
            )
            self._debug_action_logs += 1

        invalid_action = False
        action_obj: Action
        action_id: int | None = None

        if isinstance(action, Integral):
            if self._using_discrete_k():
                action_id = int(action)
                if self._action_obj_table is None or action_id < 0 or action_id >= len(self._action_obj_table):
                    if self.strict_invalid_actions:
                        raise ValueError(f"Invalid action id {action!r}")
                    action_obj = Noop()
                    action_id = self.action_spec.offsets.noop
                    invalid_action = True
                else:
                    action_obj = self._action_obj_table[action_id]
            else:
                try:
                    action_obj = unflatten(int(action), self.action_spec)
                    action_id = int(action)
                except Exception as exc:
                    if self.strict_invalid_actions:
                        raise ValueError(f"Invalid action id {action!r}") from exc
                    action_obj = Noop()
                    action_id = self.action_spec.offsets.noop
                    invalid_action = True
        else:
            action_obj = action
            if self._using_discrete_k():
                if self._action_id_by_action is None:
                    if self.strict_invalid_actions:
                        raise ValueError(f"Invalid action {action_obj!r}")
                    action_obj = Noop()
                    action_id = self.action_spec.offsets.noop
                    invalid_action = True
                else:
                    action_id = self._action_id_by_action.get(action_obj)
                    if action_id is None:
                        if self.strict_invalid_actions:
                            raise ValueError(f"Invalid action {action_obj!r}")
                        action_obj = Noop()
                        action_id = self.action_spec.offsets.noop
                        invalid_action = True
            else:
                try:
                    action_id = flatten(action_obj, self.action_spec)
                except Exception as exc:
                    if self.strict_invalid_actions:
                        raise ValueError(f"Invalid action {action_obj!r}") from exc
                    action_obj = Noop()
                    action_id = self.action_spec.offsets.noop
                    invalid_action = True

        if self._last_action_mask is None:
            raise RuntimeError("Action mask cache missing; reset() must be called first")
        mask_before = self._last_action_mask
        if self.debug_actions and mask_before is not None:
            self._debug_update_mask_stats(mask_before)
        build_action_limit_reached = self._build_action_limit_reached()
        if action_id is None or action_id >= len(mask_before) or not bool(mask_before[action_id]):
            if self.strict_invalid_actions:
                raise ValueError(f"Action not valid in current state: {action_obj!r}")
            invalid_action = True
            action_obj = Noop()
            action_id = self.action_spec.offsets.noop

        forced_start_wave = False
        if build_action_limit_reached and self.action_spec is not None:
            start_idx = self.action_spec.offsets.start_wave
            if 0 <= start_idx < len(mask_before) and bool(mask_before[start_idx]):
                if action_id != start_idx:
                    if self.strict_invalid_actions:
                        raise ValueError("StartWave required after build action limit")
                    action_obj = StartWave()
                    action_id = start_idx
                    invalid_action = True
                    forced_start_wave = True

        if self.debug_actions and mask_before is not None:
            if isinstance(action_obj, StartWave) or self._step_count == 1:
                self._debug_validate_mask(mask_before, state=self.engine.state, phase=self.phase)

        prev_state = reward_state_from(self.engine.state)
        info: dict[str, Any] = {"invalid_action": invalid_action}
        if build_action_limit_reached:
            info["build_action_limit_reached"] = True
        build_action_limit_violation = build_action_limit_reached and not isinstance(action_obj, StartWave)
        if build_action_limit_violation:
            info["build_action_limit_violation"] = True
        if forced_start_wave:
            info["action_forced_start_wave"] = True

        if self.debug_actions:
            self._debug_record_action(
                action_obj,
                invalid_action=invalid_action,
                build_action_limit_violation=build_action_limit_violation,
                forced_start_wave=forced_start_wave,
            )

        self._last_action_id = action_id
        if isinstance(action_obj, StartWave):
            apply_start = time.perf_counter() if self.timing_enabled else 0.0
            self._apply_action(action_obj)
            if self.timing_enabled:
                self.timing["apply_action_time_total"] = self.timing.get("apply_action_time_total", 0.0) + (
                    time.perf_counter() - apply_start
                )
            self._current_wave_actions.append(action_obj)
            self.episode_actions.append(self._current_wave_actions)
            self._current_wave_actions = []
            build_actions_before_wave = self.build_actions_since_wave
            wave_ticks, timeout = self._run_wave()
            info["wave_ticks"] = wave_ticks
            info["timeout"] = timeout
            if self.log_every_wave:
                state = self.engine.state
                self._log_line(
                    "wave_done wave={} ticks={} timeout={} lives={} score={} bank={}".format(
                        int(getattr(state, "level", 0)),
                        int(wave_ticks),
                        bool(timeout),
                        int(getattr(state, "lives", 0)),
                        int(getattr(state, "score", 0)),
                        int(getattr(state, "bank", 0)),
                    )
                )
            if self.debug_actions:
                state = self.engine.state
                self._debug_log_wave_summary(
                    wave=int(getattr(state, "level", 0)),
                    build_actions=build_actions_before_wave,
                    max_build_actions=self.max_build_actions,
                    limit_reached=build_action_limit_reached,
                )
                self._debug_reset_wave_stats()
            phase_transition = "WAVE_COMPLETE"
            self.build_actions_since_wave = 0
        else:
            apply_start = time.perf_counter() if self.timing_enabled else 0.0
            self._apply_action(action_obj)
            if self.timing_enabled:
                self.timing["apply_action_time_total"] = self.timing.get("apply_action_time_total", 0.0) + (
                    time.perf_counter() - apply_start
                )
            self._current_wave_actions.append(action_obj)
            phase_transition = "BUILD_ACTION"
            self.build_actions_since_wave += 1

        new_state = reward_state_from(self.engine.state)
        if self.debug_actions:
            self._debug_check_ranges(new_state)
        episode_done = self.phase == "DONE"
        truncated = False
        done = episode_done or truncated
        game_won = bool(getattr(self.engine.state, "game_won", False))
        waves_cleared = int(new_state.level)
        if done and not game_won and waves_cleared > 0:
            waves_cleared -= 1
        reward_start = time.perf_counter() if self.timing_enabled else 0.0
        breakdown = compute_reward_breakdown(
            prev_state,
            new_state,
            config=self.reward_config,
            episode_done=done,
            game_won=game_won,
        )
        reward = float(breakdown["total"])
        if self.timing_enabled:
            self.timing["reward_time_total"] = self.timing.get("reward_time_total", 0.0) + (
                time.perf_counter() - reward_start
            )
        info["r_life"] = breakdown["r_life"]
        info["r_wave"] = breakdown["r_wave"]
        info["r_terminal"] = breakdown["r_terminal"]
        info["lives"] = int(new_state.lives)
        info["wave"] = waves_cleared
        info["done"] = bool(done)
        if self.debug_actions:
            info["reward_breakdown"] = breakdown
            if phase_transition == "WAVE_COMPLETE" or done:
                self._log_line(
                    "debug_reward phase={} total={} r_life={} r_wave={} r_terminal={} delta_lives={} "
                    "delta_waves={} done={} win={}".format(
                        phase_transition,
                        breakdown["total"],
                        breakdown["r_life"],
                        breakdown["r_wave"],
                        breakdown["r_terminal"],
                        breakdown["delta_lives"],
                        breakdown["delta_waves"],
                        bool(done),
                        bool(game_won),
                    )
                )

        self._prev_lives = new_state.lives
        self._prev_wave = new_state.level
        self._episode_return += float(reward)

        terminated = episode_done
        mask_after = self._compute_action_mask()
        self._last_action_mask = mask_after
        obs_start = time.perf_counter() if self.timing_enabled else 0.0
        obs_dict = build_observation(
            self.engine.state,
            self.map_data,
            self.action_spec,
            place_candidate_cells=self._place_candidate_cells,
            place_candidate_centers=self._place_candidate_centers,
            place_candidate_static=self._place_candidate_static,
            place_candidate_tower_idx=self._place_candidate_tower_idx,
        )
        if self.timing_enabled:
            self.timing["obs_time_total"] = self.timing.get("obs_time_total", 0.0) + (
                time.perf_counter() - obs_start
            )
        obs_dict["phase"] = self.phase
        slot_size = len(obs_dict.get("tower_slot_features", []) or [])
        obs = np.asarray(
            flatten_observation(obs_dict, max_towers=self.max_towers, slot_size=slot_size),
            dtype=np.float32,
        )
        self._last_obs_dict = obs_dict
        if self.include_action_mask_in_obs:
            obs_dict["action_mask"] = mask_after
        self._debug_check_finite(kind="obs", array=obs)
        self._debug_check_finite(kind="mask", array=mask_after)
        info["action_mask"] = mask_after
        if self.timing_enabled:
            self.timing["step_calls"] = self.timing.get("step_calls", 0.0) + 1.0
            self.timing["step_time_total"] = self.timing.get("step_time_total", 0.0) + (
                time.perf_counter() - step_start
            )
        if done:
            state = self.engine.state
            self._log_line(
                "episode_done wave={} lives={} score={} bank={} game_won={}".format(
                    waves_cleared,
                    int(getattr(state, "lives", 0)),
                    int(getattr(state, "score", 0)),
                    int(getattr(state, "bank", 0)),
                    bool(getattr(state, "game_won", False)),
                )
            )
            info["episode_summary"] = {
                "total_reward": float(self._episode_return),
                "waves_cleared": waves_cleared,
                "lives_end": int(new_state.lives),
                "win": bool(game_won),
            }
            if self.debug_actions:
                self._debug_log_episode_summary(wave=int(getattr(state, "level", 0)))
        else:
            self._maybe_log_heartbeat()
        return obs, reward, terminated, truncated, info

    def _apply_discrete_k_action(self, action: Action) -> None:
        if self.engine is None or self.action_spec is None:
            return
        if isinstance(action, Noop):
            return
        if isinstance(action, StartWave):
            self.engine.act("NEXT_WAVE")
            return
        if isinstance(action, Place):
            tower_types = getattr(self.action_spec, "tower_kinds", ())
            cells_by_type = getattr(self.action_spec, "cells_by_type", ())
            if action.tower_type < 0 or action.tower_type >= len(tower_types):
                return
            if action.tower_type >= len(cells_by_type):
                return
            cell_list = cells_by_type[action.tower_type]
            if action.cell < 0 or action.cell >= len(cell_list):
                return
            cell = cell_list[action.cell]
            if cell is None:
                return
            cell_x, cell_y = cell
            tower_kind = tower_types[action.tower_type]
            self.engine.act(
                "PLACE_TOWER",
                {"cell_x": int(cell_x), "cell_y": int(cell_y), "kind": tower_kind},
            )
            return
        if isinstance(action, Upgrade):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            self.engine.act(
                "UPGRADE_TOWER",
                {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        if isinstance(action, Sell):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            self.engine.act(
                "SELL_TOWER",
                {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        if isinstance(action, SetMode):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            target_modes = getattr(self.action_spec, "target_modes", ())
            if action.mode < 0 or action.mode >= len(target_modes):
                return
            mode = target_modes[action.mode]
            self.engine.act(
                "SET_TARGET_MODE",
                {"mode": mode, "cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        raise TypeError(f"Unknown action {action!r}")

    def _apply_action(self, action: Action) -> None:
        if self.engine is None or self.action_spec is None:
            return
        if self._using_discrete_k():
            self._apply_discrete_k_action(action)
            return
        if isinstance(action, Noop):
            return
        if isinstance(action, StartWave):
            self.engine.act("NEXT_WAVE")
            return
        if isinstance(action, Place):
            if not self.action_spec.cell_positions:
                return
            if action.tower_type < 0 or action.tower_type >= len(self.action_spec.tower_kinds):
                return
            if action.cell < 0 or action.cell >= len(self.action_spec.cell_positions):
                return
            cell_x, cell_y = self.action_spec.cell_positions[action.cell]
            tower_kind = self.action_spec.tower_kinds[action.tower_type]
            self.engine.act(
                "PLACE_TOWER",
                {"cell_x": cell_x, "cell_y": cell_y, "kind": tower_kind},
            )
            return
        if isinstance(action, Upgrade):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            self.engine.act(
                "UPGRADE_TOWER",
                {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        if isinstance(action, Sell):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            self.engine.act(
                "SELL_TOWER",
                {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        if isinstance(action, SetMode):
            tower = self._resolve_tower_slot(action.tower_id)
            if tower is None:
                return
            if action.mode < 0 or action.mode >= len(self.action_spec.target_modes):
                return
            mode = self.action_spec.target_modes[action.mode]
            self.engine.act(
                "SET_TARGET_MODE",
                {"mode": mode, "cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
            )
            return
        raise TypeError(f"Unknown action {action!r}")

    def _resolve_tower_slot(self, tower_id: int):
        if self.engine is None:
            return None
        if tower_id < 0 or tower_id >= self.max_towers:
            return None
        tower_slots = get_tower_slots(self.engine.state, self.max_towers)
        return tower_slots[tower_id]

    def _is_set_mode_effective(self, action: SetMode) -> bool:
        if self.action_spec is None:
            return False
        tower = self._resolve_tower_slot(action.tower_id)
        if tower is None:
            return False
        if action.mode < 0 or action.mode >= len(self.action_spec.target_modes):
            return False
        mode_mask = self.action_spec.kind_to_mode_mask.get(str(getattr(tower, "kind", "")))
        if mode_mask is not None and action.mode < len(mode_mask) and not mode_mask[action.mode]:
            return False
        desired_mode = self.action_spec.target_modes[action.mode]
        current_mode = str(getattr(tower, "target_mode", ""))
        return desired_mode != current_mode

    def _run_wave(self) -> tuple[int, bool]:
        if self.engine is None:
            return 0, False
        self.phase = "WAVE"
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        ticks = 0
        timeout = False
        while ticks < self.max_wave_ticks:
            if getattr(self.engine.state, "game_over", False):
                break
            if not getattr(self.engine.state, "creeps", []):
                break
            self.engine.step(self.engine.FRAME_DT)
            ticks += 1
        if ticks >= self.max_wave_ticks:
            timeout = True
            logger.error("Wave simulation exceeded max_wave_ticks=%s", self.max_wave_ticks)
        if getattr(self.engine.state, "game_over", False) or timeout:
            self.phase = "DONE"
        else:
            self.phase = "BUILD"
        if self.timing_enabled:
            self.timing["wave_sim_calls"] = self.timing.get("wave_sim_calls", 0.0) + 1.0
            self.timing["wave_sim_time_total"] = self.timing.get("wave_sim_time_total", 0.0) + (
                time.perf_counter() - start_time
            )
            self.timing["wave_sim_ticks_total"] = self.timing.get("wave_sim_ticks_total", 0.0) + float(ticks)
        return ticks, timeout

    def get_timing_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"env": dict(self.timing)}
        if self.engine is not None:
            snapshot["engine"] = dict(getattr(self.engine, "timing", {}) or {})
        return snapshot

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        return None
