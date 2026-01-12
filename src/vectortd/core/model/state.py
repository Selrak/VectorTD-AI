from __future__ import annotations
from dataclasses import dataclass, field
from .entities import Creep, Tower, PulseShot, RocketShot


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
    auto_level: bool = False

    base_worth: int = 3
    base_hp: int = 550
    cc: int = 1
    rng_state: int = 1
    rng_calls: int = 0

    creeps: list[Creep] = field(default_factory=list)
    towers: list[Tower] = field(default_factory=list)
    pulses: list[PulseShot] = field(default_factory=list)
    rockets: list[RocketShot] = field(default_factory=list)
    game_over: bool = False
    game_won: bool = False
    last_wave_type: int | None = None
    last_wave_hp: int = 0
