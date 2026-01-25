from __future__ import annotations
from dataclasses import dataclass, field


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
    alive: bool = field(default=True, compare=False)


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
    target_mode: str = "closest"
    rof: int = 0
    cooldown: float = 0.0
    retarget_delay: int = 10
    target: Creep | None = None
    shot_timer: float = 0.0
    shot_x: float = 0.0
    shot_y: float = 0.0
    shot_segments: list[tuple[float, float, float, float]] = field(default_factory=list)
    shot_opacity: int = 200
    damage_buff_pct: int = 0
    range_buff_pct: int = 0
    buffed_damage: float | None = None
    buffed_range: float | None = None


@dataclass(slots=True)
class PulseShot:
    tower: Tower
    target: Creep
    from_x: float
    from_y: float
    damage: float
    alpha: float
    slow: bool = False


@dataclass(slots=True)
class RocketShot:
    kind: str
    x: float
    y: float
    target: Creep | None
    speed: float
    damage: float
