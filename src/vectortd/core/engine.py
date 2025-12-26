# src/vectortd/core/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .model.entities import Tower
from .rules.wave_spawner import start_next_wave
from .rules.creep_motion import step_creeps
from .rules.placement import place_tower


ActionType = Literal[
    "NEXT_WAVE",
    "PAUSE_TOGGLE",
    "PLACE_TOWER",
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


@dataclass(slots=True)
class Creep:
    # Position
    x: float
    y: float

    # Type (1..7 ; 8 est “mixée” dans le spawner, donc pas stocké ici)
    type_id: int

    # Mouvement sur chemin
    path: list[int]                # liste d’IDs de markers, ex: [1,3,5,...]
    path_point: int                # index dans path, cible courante = path[path_point]
    targ_x: float
    targ_y: float
    xval: float                    # direction normalisée vers targ
    yval: float

    # Stats
    worth: int
    speed: float
    max_speed: float
    hp: float
    maxhp: float


@dataclass(slots=True)
class GameState:
    # Variables globales (observées dans le SWF)
    bank: int = 250
    interest: int = 3
    level: int = 0          # numéro de vague (minuscule dans le SWF)
    lives: int = 20
    score: int = 0
    bonus_every: int = 5
    paused: bool = False

    base_worth: int = 3
    base_hp: int = 550

    # Pour la vague “mixée”
    cc: int = 1

    # Entités
    creeps: list[Creep] = field(default_factory=list)
    towers: list[Tower] = field(default_factory=list)

    # Flags
    game_over: bool = False


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

    def reset(self) -> None:
        self.state = GameState()
        self.state.base_hp = 550 if self.map.level_index < 5 else 650
        self._accum = 0.0

    def act(self, action_type: ActionType, payload: dict[str, Any] | None = None) -> None:
        if self.state.game_over:
            return

        if action_type == "PAUSE_TOGGLE":
            self.state.paused = not self.state.paused
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

        raise ValueError(f"Unknown action_type={action_type!r}")

    def step(self, dt_seconds: float) -> str | None:
        """
        Avance la simulation en "fixed-step" (équivalent frames Flash).
        """
        if self.state.game_over or self.state.paused:
            return None

        self._accum += max(0.0, dt_seconds)
        while self._accum >= self.FRAME_DT:
            self._accum -= self.FRAME_DT
            step_creeps(self.state, self.map, dt_scale=1.0)

            if self.state.lives < 1:
                self.state.game_over = True
                return "game lost"
        return None

    def observe(self) -> dict[str, Any]:
        """
        Observation minimale IA-friendly (à enrichir ensuite).
        """
        s = self.state
        return {
            "bank": s.bank,
            "lives": s.lives,
            "score": s.score,
            "wave": s.level,
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
                    "cell_x": t.cell_x,
                    "cell_y": t.cell_y,
                    "kind": t.kind,
                    "title": t.title,
                    "level": t.level,
                    "cost": t.cost,
                    "range": t.range,
                    "damage": t.damage,
                    "description": t.description,
                }
                for t in s.towers
            ],
            "game_over": s.game_over,
        }
