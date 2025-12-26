from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TowerDef:
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


TOWER_DEFS: dict[str, TowerDef] = {
    "green": TowerDef(
        kind="green",
        title="GREEN LASER 1",
        level=1,
        cost=100,
        range=70,
        damage=22,
        description=(
            "150% damage to green\n"
            "50% to red vectoids\n"
            "\n"
            "The Green laser tower fires a medium powered beam; "
            "the beam does constant damage while focused on the enemy. "
            "The range of the beam is also impressive."
        ),
        base_cost=100,
        base_range=70,
        base_damage=22,
    ),
}


def get_tower_def(kind: str) -> TowerDef:
    try:
        return TOWER_DEFS[kind]
    except KeyError as exc:
        raise KeyError(f"Unknown tower kind: {kind!r}") from exc
