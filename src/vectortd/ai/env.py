from __future__ import annotations

from pathlib import Path
import logging
import time
from typing import Any

from vectortd.core.engine import Engine
from vectortd.core.model.map import load_map_json
from vectortd.core.rng import seed_state

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
from .masking import compute_action_mask
from .obs import build_observation
from .rewards import RewardConfig, compute_reward, reward_state_from


logger = logging.getLogger(__name__)


def _resolve_map_path(map_path: str) -> Path:
    p = Path(map_path)
    if p.suffix:
        return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p
    if p.parent == Path("."):
        return Path(__file__).resolve().parents[3] / "data/maps" / f"{p.name}.json"
    return Path(__file__).resolve().parents[3] / p.with_suffix(".json")


class VectorTDEventEnv:
    def __init__(
        self,
        *,
        default_map: str = "switchback",
        max_towers: int = MAX_TOWERS,
        max_cells: int = MAX_CELLS,
        max_wave_ticks: int = 20000,
        max_build_actions: int | None = 100,
        strict_invalid_actions: bool = False,
        reward_config: RewardConfig | None = None,
        build_step_penalty: float = 0.0,
        no_life_loss_bonus: float = 50.0,
        terminal_win_bonus: float = 10_000.0,
        terminal_loss_penalty: float = 10_000.0,
        build_action_limit_penalty: float = -1.0,
        invalid_action_penalty: float = 0.0,
        include_action_mask_in_obs: bool = False,
        timing_enabled: bool = False,
    ) -> None:
        self.default_map = default_map
        self.max_towers = max_towers
        self.max_cells = max_cells
        self.max_wave_ticks = max_wave_ticks
        self.max_build_actions = max_build_actions
        self.strict_invalid_actions = strict_invalid_actions
        self.include_action_mask_in_obs = include_action_mask_in_obs
        if reward_config is None:
            reward_config = RewardConfig(
                build_step_penalty=build_step_penalty,
                no_life_loss_bonus=no_life_loss_bonus,
                terminal_win_bonus=terminal_win_bonus,
                terminal_loss_penalty=terminal_loss_penalty,
                build_action_limit_penalty=build_action_limit_penalty,
                invalid_action_penalty=invalid_action_penalty,
            )
        self.reward_config = reward_config

        self.timing_enabled = timing_enabled
        self.timing: dict[str, float] = {}

        self.engine: Engine | None = None
        self.map_data = None
        self.action_spec = None
        self.phase = "BUILD"

        self.prev_score = 0
        self.prev_lives = 0
        self.prev_bank = 0

        self.episode_actions: list[list[Action]] = []
        self._current_wave_actions: list[Action] = []
        self.episode_seed: int | None = None
        self.map_id: str | None = None
        self.map_path: str | None = None
        self.build_actions_since_wave = 0

    def reset(self, *, map_path: str | None = None, seed: int | None = None) -> dict[str, Any]:
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        map_path = map_path or self.default_map
        resolved = _resolve_map_path(map_path)
        self.map_path = str(resolved)
        self.map_data = load_map_json(resolved)
        self.map_id = str(getattr(self.map_data, "name", ""))

        self.engine = Engine(self.map_data)
        self.engine.timing_enabled = self.timing_enabled
        self.engine.reset()
        self.episode_seed = seed_state(self.engine.state, seed)

        self.action_spec = action_space_spec(
            self.map_data,
            max_towers=self.max_towers,
            max_cells=self.max_cells,
        )

        self.phase = "BUILD"
        self.prev_score = int(getattr(self.engine.state, "score", 0))
        self.prev_lives = int(getattr(self.engine.state, "lives", 0))
        self.prev_bank = int(getattr(self.engine.state, "bank", 0))

        self.episode_actions = []
        self._current_wave_actions = []
        self.build_actions_since_wave = 0

        obs = build_observation(self.engine.state, self.map_data, self.action_spec)
        obs["phase"] = self.phase
        if self.include_action_mask_in_obs:
            obs["action_mask"] = self.get_action_mask()
        if self.timing_enabled:
            self.timing["reset_calls"] = self.timing.get("reset_calls", 0.0) + 1.0
            self.timing["reset_time_total"] = self.timing.get("reset_time_total", 0.0) + (
                time.perf_counter() - start_time
            )
        return obs

    def get_action_mask(self):
        if self.engine is None or self.map_data is None or self.action_spec is None:
            raise RuntimeError("Environment not reset")
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        mask = compute_action_mask(
            self.engine.state,
            self.engine,
            self.map_data,
            self.action_spec,
            phase=self.phase,
        )
        if self.timing_enabled:
            self.timing["mask_calls"] = self.timing.get("mask_calls", 0.0) + 1.0
            self.timing["mask_time_total"] = self.timing.get("mask_time_total", 0.0) + (
                time.perf_counter() - start_time
            )
        if self._build_action_limit_reached():
            mask_len = len(mask)
            forced = [False] * mask_len
            start_idx = self.action_spec.offsets.start_wave
            if 0 <= start_idx < mask_len:
                forced[start_idx] = bool(mask[start_idx])
            try:
                import numpy as np  # type: ignore
            except Exception:
                return forced
            return np.asarray(forced, dtype=bool)
        return mask

    def _build_action_limit_reached(self) -> bool:
        if self.max_build_actions is None:
            return False
        return self.build_actions_since_wave >= self.max_build_actions

    def step(self, action: Action | int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.engine is None or self.map_data is None or self.action_spec is None:
            raise RuntimeError("Environment not reset")
        if self.phase != "BUILD":
            raise RuntimeError(f"step() called in phase={self.phase!r}")
        step_start = time.perf_counter() if self.timing_enabled else 0.0

        invalid_action = False
        action_obj: Action
        action_id: int | None = None

        if isinstance(action, int):
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
            try:
                action_id = flatten(action_obj, self.action_spec)
            except Exception as exc:
                if self.strict_invalid_actions:
                    raise ValueError(f"Invalid action {action_obj!r}") from exc
                action_obj = Noop()
                action_id = self.action_spec.offsets.noop
                invalid_action = True

        mask_before = self.get_action_mask()
        if action_id is None or action_id >= len(mask_before) or not bool(mask_before[action_id]):
            if self.strict_invalid_actions:
                raise ValueError(f"Action not valid in current state: {action_obj!r}")
            invalid_action = True
            action_obj = Noop()
            action_id = self.action_spec.offsets.noop

        prev_state = reward_state_from(self.engine.state)
        info: dict[str, Any] = {"invalid_action": invalid_action}
        build_action_limit_reached = self._build_action_limit_reached()
        if build_action_limit_reached:
            info["build_action_limit_reached"] = True
        build_action_limit_violation = build_action_limit_reached and not isinstance(action_obj, StartWave)
        if build_action_limit_violation:
            info["build_action_limit_violation"] = True

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
            wave_ticks, timeout = self._run_wave()
            info["wave_ticks"] = wave_ticks
            info["timeout"] = timeout
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
        episode_done = self.phase == "DONE"
        game_won = bool(getattr(self.engine.state, "game_won", False))
        reward_start = time.perf_counter() if self.timing_enabled else 0.0
        reward = compute_reward(
            prev_state,
            new_state,
            phase_transition=phase_transition,
            config=self.reward_config,
            invalid_action=invalid_action,
            build_action_limit_violation=build_action_limit_violation,
            episode_done=episode_done,
            game_won=game_won,
        )
        if self.timing_enabled:
            self.timing["reward_time_total"] = self.timing.get("reward_time_total", 0.0) + (
                time.perf_counter() - reward_start
            )

        self.prev_score = new_state.score
        self.prev_lives = new_state.lives
        self.prev_bank = new_state.bank

        done = episode_done
        mask_after = self.get_action_mask()
        obs_start = time.perf_counter() if self.timing_enabled else 0.0
        obs = build_observation(self.engine.state, self.map_data, self.action_spec)
        if self.timing_enabled:
            self.timing["obs_time_total"] = self.timing.get("obs_time_total", 0.0) + (
                time.perf_counter() - obs_start
            )
        obs["phase"] = self.phase
        if self.include_action_mask_in_obs:
            obs["action_mask"] = mask_after
        info["action_mask"] = mask_after
        if self.timing_enabled:
            self.timing["step_calls"] = self.timing.get("step_calls", 0.0) + 1.0
            self.timing["step_time_total"] = self.timing.get("step_time_total", 0.0) + (
                time.perf_counter() - step_start
            )
        return obs, reward, done, info

    def _apply_action(self, action: Action) -> None:
        if self.engine is None or self.action_spec is None:
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
        return None
