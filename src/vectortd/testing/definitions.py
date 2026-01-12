from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from vectortd.core.model.state import GameState
from vectortd.core.rules.placement import place_tower, upgrade_tower
from vectortd.core.rules.wave_spawner import start_next_wave


TESTS_DIRNAME = Path("data/tests")
DEFAULT_MAX_TICKS = 6000


@dataclass(frozen=True, slots=True)
class TowerSpec:
    kind: str
    cell_x: int
    cell_y: int
    upgrades: int


@dataclass(frozen=True, slots=True)
class TestGoal:
    type: str
    max_ticks: int
    min_lives: int | None = None


@dataclass(frozen=True, slots=True)
class TestDefinition:
    name: str
    map: str
    starting_bank: int
    starting_wave: int
    starting_lives: int
    towers: tuple[TowerSpec, ...]
    goal: TestGoal


class TestRunner:
    def __init__(self, definition: TestDefinition, state: GameState) -> None:
        self.definition = definition
        self.state = state
        self.ticks = 0.0
        self.status = "running"

    def advance(self, steps: float) -> None:
        if self.status != "running":
            return
        self.ticks += float(steps)
        self._evaluate()

    def _evaluate(self) -> None:
        goal = self.definition.goal
        if goal.type != "clear_wave":
            raise ValueError(f"Unsupported test goal type {goal.type!r}")
        min_lives = goal.min_lives
        if min_lives is None:
            min_lives = self.definition.starting_lives
        if self.state.lives < min_lives:
            self.status = "failed"
            return
        if self.ticks >= goal.max_ticks:
            self.status = "failed"
            return
        if not self.state.creeps:
            self.status = "passed"


def resolve_tests_dir(root: Path) -> Path:
    return root / TESTS_DIRNAME


def resolve_map_path(map_arg: str, root: Path) -> Path:
    p = Path(map_arg)
    if p.suffix:
        return p if p.is_absolute() else root / p
    if p.parent == Path("."):
        return root / "data/maps" / f"{p.name}.json"
    return root / p.with_suffix(".json")


def load_test_definitions(test_dir: Path) -> dict[str, TestDefinition]:
    if not test_dir.exists():
        return {}
    tests: dict[str, TestDefinition] = {}
    for path in sorted(test_dir.glob("*.json")):
        tests[path.stem] = load_test_definition(path)
    return tests


def load_test_definition(path: Path) -> TestDefinition:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    name = str(data.get("name") or path.stem)
    map_name = data.get("map")
    if not map_name:
        raise ValueError(f"Missing 'map' in test definition {path}")
    starting_bank = data.get("starting_money", data.get("starting_bank"))
    if starting_bank is None:
        raise ValueError(f"Missing 'starting_money' in test definition {path}")
    starting_wave = data.get("starting_wave")
    if starting_wave is None:
        raise ValueError(f"Missing 'starting_wave' in test definition {path}")
    starting_lives = data.get("starting_lives")
    if starting_lives is None:
        raise ValueError(f"Missing 'starting_lives' in test definition {path}")

    towers: list[TowerSpec] = []
    for tower in data.get("towers", []):
        kind = tower.get("kind")
        if not kind:
            raise ValueError(f"Missing tower 'kind' in test definition {path}")
        cell = tower.get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError(f"Invalid tower 'cell' in test definition {path}: {cell!r}")
        upgrades = int(tower.get("upgrades", 0))
        towers.append(
            TowerSpec(
                kind=str(kind),
                cell_x=int(cell[0]),
                cell_y=int(cell[1]),
                upgrades=upgrades,
            )
        )

    goal = data.get("goal") or {}
    goal_type = str(goal.get("type", "clear_wave"))
    max_ticks = int(goal.get("max_ticks", DEFAULT_MAX_TICKS))
    min_lives = goal.get("min_lives")
    if min_lives is not None:
        min_lives = int(min_lives)

    return TestDefinition(
        name=name,
        map=str(map_name),
        starting_bank=int(starting_bank),
        starting_wave=int(starting_wave),
        starting_lives=int(starting_lives),
        towers=tuple(towers),
        goal=TestGoal(type=goal_type, max_ticks=max_ticks, min_lives=min_lives),
    )


def apply_test_definition(state: GameState, map_data, definition: TestDefinition) -> None:
    state.bank = int(definition.starting_bank)
    state.lives = int(definition.starting_lives)
    state.level = max(0, int(definition.starting_wave) - 1)
    state.score = 0
    state.paused = False
    state.auto_level = False
    state.game_over = False
    state.game_won = False
    state.last_wave_type = None
    state.last_wave_hp = 0
    state.creeps.clear()
    state.towers.clear()
    state.pulses.clear()
    state.rockets.clear()

    for tower_spec in definition.towers:
        tower = place_tower(
            state,
            map_data,
            tower_spec.cell_x,
            tower_spec.cell_y,
            tower_spec.kind,
        )
        if tower is None:
            raise ValueError(
                "Test tower placement failed "
                f"kind={tower_spec.kind} cell=({tower_spec.cell_x},{tower_spec.cell_y})"
            )
        for _ in range(max(0, tower_spec.upgrades)):
            if not upgrade_tower(state, tower):
                raise ValueError(
                    "Test tower upgrade failed "
                    f"kind={tower_spec.kind} level={tower.level}"
                )

    if definition.starting_wave > 0:
        start_next_wave(state, map_data)
