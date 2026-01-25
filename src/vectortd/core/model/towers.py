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
    rof: int
    target_mode: str
    target_modes: tuple[str, ...]
    description: str
    base_cost: int
    base_range: int
    base_damage: int
    tag_color: tuple[int, int, int]


TARGET_MODES_STANDARD: tuple[str, ...] = ("closest", "hardest", "weakest")
TARGET_MODES_RANDOM: tuple[str, ...] = ("random",)
TARGET_MODES_FASTEST: tuple[str, ...] = ("fastest",)
BUFF_TOWER_KINDS: tuple[str, ...] = ("buffD", "buffR")
BUFF_TOWER_UPS_COST = 1


def is_buff_tower(kind: str) -> bool:
    return str(kind) in BUFF_TOWER_KINDS


TOWER_DEFS: dict[str, TowerDef] = {
    "green": TowerDef(
        kind="green",
        title="GREEN LASER 1",
        level=1,
        cost=100,
        range=70,
        damage=22,
        rof=0,
        target_mode="closest",
        target_modes=TARGET_MODES_STANDARD,
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
        tag_color=(0, 220, 0),
    ),
    "green2": TowerDef(
        kind="green2",
        title="GREEN LASER 2",
        level=1,
        cost=400,
        range=70,
        damage=45,
        rof=0,
        target_mode="closest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to green\n"
            "50% to red vectoids\n"
            "Single bounce\n"
            "\n"
            "A stronger beam than green laser 1 which will damage 2 vectoids "
            "at a time, both targets take equal damage."
        ),
        base_cost=400,
        base_range=70,
        base_damage=45,
        tag_color=(0, 220, 0),
    ),
    "green3": TowerDef(
        kind="green3",
        title="GREEN LASER 3",
        level=1,
        cost=2000,
        range=70,
        damage=180,
        rof=0,
        target_mode="closest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to green\n"
            "50% to red vectoids\n"
            "Bounces twice\n"
            "\n"
            "A stronger beam than green laser 2 which will damage 3 vectoids "
            "at a time, all 3 targets take equal damage."
        ),
        base_cost=2000,
        base_range=70,
        base_damage=180,
        tag_color=(0, 220, 0),
    ),
    "red_refractor": TowerDef(
        kind="red_refractor",
        title="RED REFRACTOR",
        level=1,
        cost=200,
        range=80,
        damage=110,
        rof=10,
        target_mode="closest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to red\n"
            "50% to green vectoids\n"
            "Splash damage\n"
            "\n"
            "Red laser is lower power than green, however uses a frequency "
            "which causes it to reflect and refract off the enemy vectoid "
            "damaging all those around it."
        ),
        base_cost=200,
        base_range=80,
        base_damage=110,
        tag_color=(220, 0, 0),
    ),
    "red_spammer": TowerDef(
        kind="red_spammer",
        title="LITTLE RED SPAMMER",
        level=1,
        cost=800,
        range=80,
        damage=400,
        rof=4,
        target_mode="random",
        target_modes=TARGET_MODES_RANDOM,
        description=(
            "150% damage to red\n"
            "50% to green vectoids\n"
            "Multiple random targets\n"
            "\n"
            "Launches small heat seaking rockets at random vectoids in range. "
            "The rockets have no onboard AI so if the target is destroyed the "
            "rocket self destructs."
        ),
        base_cost=800,
        base_range=80,
        base_damage=400,
        tag_color=(220, 0, 0),
    ),
    "red_rockets": TowerDef(
        kind="red_rockets",
        title="RED ROCKETS",
        level=1,
        cost=2500,
        range=150,
        damage=30000,
        rof=45,
        target_mode="hardest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to red\n"
            "50% to green vectoids\n"
            "Two targets at a time\n"
            "\n"
            "Fires 2 heat seaking rockets which target random vectoids. If the "
            "target is destroyed the rockets AI selects the next closest "
            "vectoid and changes course."
        ),
        base_cost=2500,
        base_range=150,
        base_damage=30000,
        tag_color=(220, 0, 0),
    ),
    "purple1": TowerDef(
        kind="purple1",
        title="PURPLE POWER 1",
        level=1,
        cost=300,
        range=100,
        damage=2650,
        rof=40,
        target_mode="hardest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to purple\n"
            "50% to blue vectoids\n"
            "\n"
            "After targeting an enemy vectoid it drains power. Once the weapon "
            "reaches critical mass it feeds the energy back into the target in "
            "a split second, causing a large amount of damage."
        ),
        base_cost=300,
        base_range=100,
        base_damage=2650,
        tag_color=(190, 0, 190),
    ),
    "purple2": TowerDef(
        kind="purple2",
        title="PURPLE POWER 2",
        level=1,
        cost=900,
        range=100,
        damage=8500,
        rof=40,
        target_mode="hardest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to purple\n"
            "50% to blue vectoids\n"
            "\n"
            "Same tower as Purple Power 1, however this has 2 beams that "
            "focus on the same target causing more than twice the damage."
        ),
        base_cost=900,
        base_range=100,
        base_damage=8500,
        tag_color=(190, 0, 190),
    ),
    "purple3": TowerDef(
        kind="purple3",
        title="PURPLE POWER 3",
        level=1,
        cost=2800,
        range=100,
        damage=22000,
        rof=40,
        target_mode="hardest",
        target_modes=TARGET_MODES_STANDARD,
        description=(
            "150% damage to purple\n"
            "50% to blue vectoids\n"
            "Slows target\n"
            "\n"
            "This is the same as Purple Laser 1 and 2 but with the ability to "
            "drain so much power from the target it causes it to slow to a stop."
        ),
        base_cost=2800,
        base_range=100,
        base_damage=22000,
        tag_color=(190, 0, 190),
    ),
    "blue1": TowerDef(
        kind="blue1",
        title="BLUE RAYS 1",
        level=1,
        cost=300,
        range=70,
        damage=1000,
        rof=40,
        target_mode="fastest",
        target_modes=TARGET_MODES_FASTEST,
        description=(
            "150% damage to blue\n"
            "50% to purple vectoids\n"
            "\n"
            "The blue rays from this tower drain power from the enemy causing "
            "them to lose speed. The blue ray tower can target upto 4 "
            "different enemies in a single shot. It is purpose is to slow, "
            "so damage is low."
        ),
        base_cost=300,
        base_range=70,
        base_damage=1000,
        tag_color=(0, 150, 220),
    ),
    "blue2": TowerDef(
        kind="blue2",
        title="BLUE RAYS 2",
        level=1,
        cost=500,
        range=80,
        damage=6000,
        rof=120,
        target_mode="fastest",
        target_modes=TARGET_MODES_FASTEST,
        description=(
            "150% damage to blue\n"
            "50% to purple vectoids\n"
            "\n"
            "When this tower hits an enemy it removes all of its power and "
            "then some, causing the enemy to stop dead for a second or 2 "
            "while it re-boots."
        ),
        base_cost=500,
        base_range=80,
        base_damage=6000,
        tag_color=(0, 150, 220),
    ),
    "buffD": TowerDef(
        kind="buffD",
        title="BUFF DAMAGE",
        level=1,
        cost=0,
        range=80,
        damage=0,
        rof=0,
        target_mode="",
        target_modes=(),
        description="Boosts nearby towers' damage by 25%.",
        base_cost=0,
        base_range=80,
        base_damage=0,
        tag_color=(220, 180, 0),
    ),
    "buffR": TowerDef(
        kind="buffR",
        title="BUFF RANGE",
        level=1,
        cost=0,
        range=80,
        damage=0,
        rof=0,
        target_mode="",
        target_modes=(),
        description="Boosts nearby towers' range by 25%.",
        base_cost=0,
        base_range=80,
        base_damage=0,
        tag_color=(0, 180, 220),
    ),
}

TOWER_ORDER: list[str] = [
    "green",
    "red_refractor",
    "purple1",
    "blue1",
    "green2",
    "red_spammer",
    "purple2",
    "blue2",
    "green3",
    "red_rockets",
    "purple3",
]


def get_tower_def(kind: str) -> TowerDef:
    try:
        return TOWER_DEFS[kind]
    except KeyError as exc:
        raise KeyError(f"Unknown tower kind: {kind!r}") from exc


def list_tower_defs() -> list[TowerDef]:
    return [TOWER_DEFS[kind] for kind in TOWER_ORDER if kind in TOWER_DEFS]
