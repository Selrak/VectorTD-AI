from __future__ import annotations

from pathlib import Path
import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np

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
from .obs import _tower_slot_features, build_observation
from .obs_flatten import SCALAR_KEYS, flatten_observation
from .rewards import RewardConfig, compute_reward, reward_state_from


logger = logging.getLogger(__name__)


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
        log_dir: str | Path | None = None,
        log_prefix: str | None = None,
        log_interval_sec: float = 10.0,
        log_every_wave: bool = True,
        log_every_reset: bool = True,
    ) -> None:
        super().__init__()
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

    @property
    def last_obs(self) -> dict[str, Any] | None:
        return self._last_obs_dict

    def _bootstrap_spaces(self) -> None:
        try:
            resolved = _resolve_map_path(self.default_map)
            map_data = load_map_json(resolved)
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
        self.episode_seed = seed_state(self.engine.state, engine_seed)
        self._step_count = 0

        self.action_spec = action_space_spec(
            self.map_data,
            max_towers=self.max_towers,
            max_cells=self.max_cells,
        )
        self.action_space = gym.spaces.Discrete(self.action_spec.num_actions)

        self.phase = "BUILD"
        self.prev_score = int(getattr(self.engine.state, "score", 0))
        self.prev_lives = int(getattr(self.engine.state, "lives", 0))
        self.prev_bank = int(getattr(self.engine.state, "bank", 0))

        self.episode_actions = []
        self._current_wave_actions = []
        self.build_actions_since_wave = 0

        obs_dict = build_observation(self.engine.state, self.map_data, self.action_spec)
        obs_dict["phase"] = self.phase
        slot_size = len(obs_dict.get("tower_slot_features", []) or [])
        self._obs_dim = len(SCALAR_KEYS) + self.max_towers * slot_size
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
        start_time = time.perf_counter() if self.timing_enabled else 0.0
        mask = compute_action_mask(
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
            if 0 <= start_idx < mask_len:
                forced[start_idx] = bool(mask_array[start_idx])
            return forced
        return mask_array

    def action_masks(self) -> np.ndarray:
        if self._last_action_mask is None:
            self._last_action_mask = self._compute_action_mask()
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

        if self._last_action_mask is None:
            raise RuntimeError("Action mask cache missing; reset() must be called first")
        mask_before = self._last_action_mask
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

        prev_state = reward_state_from(self.engine.state)
        info: dict[str, Any] = {"invalid_action": invalid_action}
        if build_action_limit_reached:
            info["build_action_limit_reached"] = True
        build_action_limit_violation = build_action_limit_reached and not isinstance(action_obj, StartWave)
        if build_action_limit_violation:
            info["build_action_limit_violation"] = True
        if forced_start_wave:
            info["action_forced_start_wave"] = True

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

        terminated = episode_done
        truncated = False
        mask_after = self._compute_action_mask()
        self._last_action_mask = mask_after
        obs_start = time.perf_counter() if self.timing_enabled else 0.0
        obs_dict = build_observation(self.engine.state, self.map_data, self.action_spec)
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
        info["action_mask"] = mask_after
        if self.timing_enabled:
            self.timing["step_calls"] = self.timing.get("step_calls", 0.0) + 1.0
            self.timing["step_time_total"] = self.timing.get("step_time_total", 0.0) + (
                time.perf_counter() - step_start
            )
        if terminated:
            state = self.engine.state
            self._log_line(
                "episode_done wave={} lives={} score={} bank={} game_won={}".format(
                    int(getattr(state, "level", 0)),
                    int(getattr(state, "lives", 0)),
                    int(getattr(state, "score", 0)),
                    int(getattr(state, "bank", 0)),
                    bool(getattr(state, "game_won", False)),
                )
            )
        else:
            self._maybe_log_heartbeat()
        return obs, reward, terminated, truncated, info

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
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        return None
