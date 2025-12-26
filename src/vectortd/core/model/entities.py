from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Creep:
    x: float
    y: float
    type_id: int

    path: list[int]
    path_point: int
    targ_x: float
    targ_y: float
    xval: float
    yval: float

    worth: int
    speed: float
    max_speed: float
    hp: float
    maxhp: float


@dataclass(slots=True)
class Tower:
    cell_x: int
    cell_y: int
    kind: str
    title: str
    level: int
    cost: int
    range: int
    damage: int
    description: str
    base_cost: int
    base_range: int
    base_damage: int
