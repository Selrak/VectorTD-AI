from __future__ import annotations
from dataclasses import dataclass, field
from .entities import Creep, Tower


@dataclass(slots=True)
class GameState:
    bank: int = 250
    interest: int = 3
    ups: int = 0
    level: int = 0
    lives: int = 20
    score: int = 0
    bonus_every: int = 5
    paused: bool = False

    base_worth: int = 3
    base_hp: int = 550
    cc: int = 1

    creeps: list[Creep] = field(default_factory=list)
    towers: list[Tower] = field(default_factory=list)
    game_over: bool = False
