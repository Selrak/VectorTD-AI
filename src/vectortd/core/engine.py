# src/vectortd/core/engine.py
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

from .model.state import GameState
from .model.towers import get_tower_def
from .rules.wave_spawner import maybe_auto_next_wave, start_next_wave, wave_display_info
from .rules.tower_attack import step_towers
from .rules.creep_motion import step_creeps
from .rules.placement import place_tower, sell_tower, set_target_mode, upgrade_tower


ActionType = Literal[
    "NEXT_WAVE",
    "AUTO_WAVE_TOGGLE",
    "PAUSE_TOGGLE",
    "PLACE_TOWER",
    "UPGRADE_TOWER",
    "SELL_TOWER",
    "SET_TARGET_MODE",
    # À compléter ensuite :
    # "PLACE_TOWER", "UPGRADE_TOWER", "SELL_TOWER", "SET_TARGET_MODE", ...
]


@dataclass(slots=True)
class MapData:
    """
    Représentation minimale, headless-friendly.

    markers: dict[int, (x,y)] où les clés sont 1..34 pour Switchback.
    paths: list[list[int]] où chaque int est un index de marker (1..34).
    spawn_dir: "up"/"down"/"left"/"right"
    """
    name: str
    level_index: int
    width: int
    height: int
    grid: int
    spawn_dir: Literal["up", "down", "left", "right"]
    paths: list[list[int]]
    markers: dict[int, tuple[float, float]]

    def marker_xy(self, marker_id: int) -> tuple[float, float]:
        try:
            return self.markers[marker_id]
        except KeyError as e:
            raise KeyError(f"Missing marker m{marker_id} in map '{self.name}'") from e


class Engine:
    """
    Moteur déterministe : aucune dépendance GUI.
    Le pas de simulation est un "frame step" Flash-like (60 fps).
    """
    FRAME_DT = 1.0 / 60.0

    def __init__(self, map_data: MapData):
        self.map = map_data
        self.state = GameState()

        # Dans le SWF : baseHP dépend de Level (index de map).
        # Ici : même règle minimale.
        self.state.base_hp = 550 if self.map.level_index < 5 else 650

        self._accum = 0.0
        self.timing_enabled = False
        self.timing: dict[str, float] = {}
        self.tower_timing_enabled = False
        self.tower_timing: dict[str, float] = {}

    def reset(self) -> None:
        self.state = GameState()
        self.state.base_hp = 550 if self.map.level_index < 5 else 650
        self._accum = 0.0

    def _resolve_tower_from_payload(self, payload: dict[str, Any] | None):
        if not payload:
            return None
        tower_id = payload.get("tower_id")
        if tower_id is not None:
            try:
                tower_id = int(tower_id)
            except (TypeError, ValueError):
                tower_id = None
        if tower_id is not None:
            for candidate in self.state.towers:
                if id(candidate) == tower_id:
                    return candidate
        cell_x = payload.get("cell_x")
        cell_y = payload.get("cell_y")
        if cell_x is not None and cell_y is not None:
            try:
                cell_x = int(cell_x)
                cell_y = int(cell_y)
            except (TypeError, ValueError):
                return None
            for candidate in self.state.towers:
                if candidate.cell_x == cell_x and candidate.cell_y == cell_y:
                    return candidate
        return None

    def act(self, action_type: ActionType, payload: dict[str, Any] | None = None) -> None:
        if self.state.game_over:
            return

        if action_type == "PAUSE_TOGGLE":
            self.state.paused = not self.state.paused
            return

        if action_type == "AUTO_WAVE_TOGGLE":
            self.state.auto_level = not self.state.auto_level
            if self.state.auto_level:
                maybe_auto_next_wave(self.state, self.map)
            return

        if action_type == "NEXT_WAVE":
            if not self.state.paused:
                start_next_wave(self.state, self.map)
            return
        if action_type == "PLACE_TOWER":
            if not payload:
                return
            cell_x = payload.get("cell_x")
            cell_y = payload.get("cell_y")
            if cell_x is None or cell_y is None:
                return
            kind = payload.get("kind", "green")
            place_tower(self.state, self.map, int(cell_x), int(cell_y), tower_kind=str(kind))
            return
        if action_type == "UPGRADE_TOWER":
            tower = self._resolve_tower_from_payload(payload)
            upgrade_tower(self.state, tower, self.map)
            return
        if action_type == "SELL_TOWER":
            tower = self._resolve_tower_from_payload(payload)
            sell_tower(self.state, tower, self.map)
            return
        if action_type == "SET_TARGET_MODE":
            if not payload:
                return
            mode = payload.get("mode")
            if mode is None:
                return
            tower = self._resolve_tower_from_payload(payload)
            if tower is None:
                return
            set_target_mode(tower, str(mode))
            return

        raise ValueError(f"Unknown action_type={action_type!r}")

    def step(self, dt_seconds: float) -> str | None:
        """
        Avance la simulation en "fixed-step" (équivalent frames Flash).
        """
        if self.state.game_over or self.state.paused:
            return None

        if not self.timing_enabled:
            self._accum += max(0.0, dt_seconds)
            while self._accum >= self.FRAME_DT:
                self._accum -= self.FRAME_DT
                step_creeps(self.state, self.map, dt_scale=1.0)
                if self.tower_timing_enabled:
                    step_towers(
                        self.state,
                        self.map,
                        dt_scale=1.0,
                        timing=self.tower_timing,
                        timing_mode="tower_kind",
                    )
                else:
                    step_towers(self.state, self.map, dt_scale=1.0)
                maybe_auto_next_wave(self.state, self.map)
                if self.state.game_over and self.state.game_won:
                    return "WIN"

                if self.state.lives < 1:
                    self.state.game_over = True
                    self.state.game_won = False
                    return "game lost"
            return None

        start_total = time.perf_counter()
        self._accum += max(0.0, dt_seconds)
        ticks = 0
        creep_time = 0.0
        tower_time = 0.0
        auto_time = 0.0
        result = None
        while self._accum >= self.FRAME_DT:
            self._accum -= self.FRAME_DT
            ticks += 1
            t0 = time.perf_counter()
            step_creeps(self.state, self.map, dt_scale=1.0, timing=self.timing)
            creep_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            step_towers(self.state, self.map, dt_scale=1.0, timing=self.timing, timing_mode="full")
            tower_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            maybe_auto_next_wave(self.state, self.map)
            auto_time += time.perf_counter() - t0
            if self.state.game_over and self.state.game_won:
                result = "WIN"
                break
            if self.state.lives < 1:
                self.state.game_over = True
                self.state.game_won = False
                result = "game lost"
                break
        total_time = time.perf_counter() - start_total
        self.timing["step_calls"] = self.timing.get("step_calls", 0.0) + 1.0
        self.timing["step_time_total"] = self.timing.get("step_time_total", 0.0) + total_time
        self.timing["step_ticks"] = self.timing.get("step_ticks", 0.0) + float(ticks)
        self.timing["creep_time_total"] = self.timing.get("creep_time_total", 0.0) + creep_time
        self.timing["tower_time_total"] = self.timing.get("tower_time_total", 0.0) + tower_time
        self.timing["auto_wave_time_total"] = self.timing.get("auto_wave_time_total", 0.0) + auto_time
        return result

    def observe(self) -> dict[str, Any]:
        """
        Observation minimale IA-friendly (à enrichir ensuite).
        """
        s = self.state
        wave_info = wave_display_info(s, self.map)
        return {
            "bank": s.bank,
            "lives": s.lives,
            "score": s.score,
            "ups": s.ups,
            "wave": s.level,
            "wave_current": wave_info["current"],
            "wave_next": wave_info["next"],
            "paused": s.paused,
            "base_hp": s.base_hp,
            "base_worth": s.base_worth,
            "creeps": [
                {
                    "x": c.x,
                    "y": c.y,
                    "type": c.type_id,
                    "hp": c.hp,
                    "maxhp": c.maxhp,
                    "speed": c.speed,
                    "worth": c.worth,
                    "path_point": c.path_point,
                }
                for c in s.creeps
            ],
            "towers": [
                {
                    "tower_id": id(t),
                    "cell_x": t.cell_x,
                    "cell_y": t.cell_y,
                    "kind": t.kind,
                    "title": t.title,
                    "level": t.level,
                    "cost": t.cost,
                    "range": t.buffed_range if t.buffed_range is not None else t.range,
                    "damage": t.buffed_damage if t.buffed_damage is not None else t.damage,
                    "description": t.description,
                    "target_mode": t.target_mode,
                    "target_modes": list(get_tower_def(t.kind).target_modes),
                }
                for t in s.towers
            ],
            "game_over": s.game_over,
        }
