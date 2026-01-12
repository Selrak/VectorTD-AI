from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from vectortd.core.model.towers import list_tower_defs
from vectortd.core.rules.placement import buildable_cells


logger = logging.getLogger(__name__)

MAX_TOWERS = 32
MAX_CELLS = 625


class ActionType(Enum):
    START_WAVE = "START_WAVE"
    PLACE = "PLACE"
    UPGRADE = "UPGRADE"
    SELL = "SELL"
    SET_MODE = "SET_MODE"
    NOOP = "NOOP"


@dataclass(frozen=True, slots=True)
class StartWave:
    pass


@dataclass(frozen=True, slots=True)
class Place:
    tower_type: int
    cell: int


@dataclass(frozen=True, slots=True)
class Upgrade:
    tower_id: int


@dataclass(frozen=True, slots=True)
class Sell:
    tower_id: int


@dataclass(frozen=True, slots=True)
class SetMode:
    tower_id: int
    mode: int


@dataclass(frozen=True, slots=True)
class Noop:
    pass


Action = StartWave | Place | Upgrade | Sell | SetMode | Noop


@dataclass(frozen=True, slots=True)
class ActionOffsets:
    noop: int
    start_wave: int
    place: int
    upgrade: int
    sell: int
    set_mode: int


@dataclass(frozen=True, slots=True)
class ActionSpaceSpec:
    max_towers: int
    max_cells: int
    map_name: str
    tower_kinds: tuple[str, ...]
    tower_costs: tuple[int, ...]
    target_modes: tuple[str, ...]
    cell_positions: tuple[tuple[int, int], ...]
    cell_index: dict[tuple[int, int], int]
    kind_to_mode_mask: dict[str, tuple[bool, ...]]
    offsets: ActionOffsets
    place_count: int
    upgrade_count: int
    sell_count: int
    set_mode_count: int
    num_actions: int


_CAP_LOGGED: set[tuple[str, str]] = set()


def _log_cap_exceeded(kind: str, count: int, cap: int, *, context: str) -> None:
    key = (kind, context)
    if key in _CAP_LOGGED:
        return
    _CAP_LOGGED.add(key)
    logger.error("%s cap exceeded (%s > %s) for %s", kind, count, cap, context)


def action_space_spec(
    map_data,
    *,
    max_towers: int = MAX_TOWERS,
    max_cells: int = MAX_CELLS,
) -> ActionSpaceSpec:
    tower_defs = list_tower_defs()
    tower_kinds = tuple(t.kind for t in tower_defs)
    tower_costs = tuple(int(getattr(t, "cost", 0)) for t in tower_defs)

    target_modes: list[str] = []
    seen_modes: set[str] = set()
    for tower_def in tower_defs:
        for mode in tower_def.target_modes:
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            target_modes.append(mode)

    cells = list(buildable_cells(map_data))
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    if len(cells) > max_cells:
        _log_cap_exceeded("cells", len(cells), max_cells, context=str(getattr(map_data, "name", "")))
        cells = cells[:max_cells]
    cell_index = {(int(x), int(y)): idx for idx, (x, y) in enumerate(cells)}

    place_count = len(tower_kinds) * len(cells)
    upgrade_count = max_towers
    sell_count = max_towers
    set_mode_count = max_towers * len(target_modes)
    kind_to_mode_mask: dict[str, tuple[bool, ...]] = {}
    for tower_def in tower_defs:
        kind_to_mode_mask[str(tower_def.kind)] = tuple(mode in tower_def.target_modes for mode in target_modes)

    offsets = ActionOffsets(
        noop=0,
        start_wave=1,
        place=2,
        upgrade=2 + place_count,
        sell=2 + place_count + upgrade_count,
        set_mode=2 + place_count + upgrade_count + sell_count,
    )
    num_actions = offsets.set_mode + set_mode_count

    return ActionSpaceSpec(
        max_towers=max_towers,
        max_cells=max_cells,
        map_name=str(getattr(map_data, "name", "")),
        tower_kinds=tower_kinds,
        tower_costs=tower_costs,
        target_modes=tuple(target_modes),
        cell_positions=tuple((int(x), int(y)) for (x, y) in cells),
        cell_index=cell_index,
        kind_to_mode_mask=kind_to_mode_mask,
        offsets=offsets,
        place_count=place_count,
        upgrade_count=upgrade_count,
        sell_count=sell_count,
        set_mode_count=set_mode_count,
        num_actions=num_actions,
    )


def tower_sort_key(tower) -> tuple[int, int, int]:
    return (int(getattr(tower, "cell_y", 0)), int(getattr(tower, "cell_x", 0)), id(tower))


def sorted_towers(state) -> list:
    towers = list(getattr(state, "towers", []) or [])
    towers.sort(key=tower_sort_key)
    return towers


def get_tower_slots(state, max_towers: int) -> list:
    towers = sorted_towers(state)
    slots: list = [None] * max_towers
    for idx, tower in enumerate(towers[:max_towers]):
        slots[idx] = tower
    return slots


def flatten(action: Action, spec: ActionSpaceSpec) -> int:
    if isinstance(action, Noop):
        return spec.offsets.noop
    if isinstance(action, StartWave):
        return spec.offsets.start_wave
    if isinstance(action, Place):
        if action.tower_type < 0 or action.tower_type >= len(spec.tower_kinds):
            raise ValueError(f"Invalid tower_type={action.tower_type}")
        if action.cell < 0 or action.cell >= len(spec.cell_positions):
            raise ValueError(f"Invalid cell={action.cell}")
        return spec.offsets.place + action.tower_type * len(spec.cell_positions) + action.cell
    if isinstance(action, Upgrade):
        if action.tower_id < 0 or action.tower_id >= spec.max_towers:
            raise ValueError(f"Invalid tower_id={action.tower_id}")
        return spec.offsets.upgrade + action.tower_id
    if isinstance(action, Sell):
        if action.tower_id < 0 or action.tower_id >= spec.max_towers:
            raise ValueError(f"Invalid tower_id={action.tower_id}")
        return spec.offsets.sell + action.tower_id
    if isinstance(action, SetMode):
        if action.tower_id < 0 or action.tower_id >= spec.max_towers:
            raise ValueError(f"Invalid tower_id={action.tower_id}")
        if action.mode < 0 or action.mode >= len(spec.target_modes):
            raise ValueError(f"Invalid mode={action.mode}")
        return spec.offsets.set_mode + action.tower_id * len(spec.target_modes) + action.mode
    raise TypeError(f"Unsupported action {action!r}")


def unflatten(action_id: int, spec: ActionSpaceSpec) -> Action:
    if action_id == spec.offsets.noop:
        return Noop()
    if action_id == spec.offsets.start_wave:
        return StartWave()

    if spec.place_count > 0:
        place_end = spec.offsets.place + spec.place_count
        if spec.offsets.place <= action_id < place_end:
            idx = action_id - spec.offsets.place
            cell_count = len(spec.cell_positions)
            if cell_count <= 0:
                raise ValueError("No cells available for PLACE actions")
            tower_type = idx // cell_count
            cell = idx % cell_count
            return Place(tower_type=tower_type, cell=cell)

    upgrade_end = spec.offsets.upgrade + spec.upgrade_count
    if spec.offsets.upgrade <= action_id < upgrade_end:
        return Upgrade(tower_id=action_id - spec.offsets.upgrade)

    sell_end = spec.offsets.sell + spec.sell_count
    if spec.offsets.sell <= action_id < sell_end:
        return Sell(tower_id=action_id - spec.offsets.sell)

    set_mode_end = spec.offsets.set_mode + spec.set_mode_count
    if spec.offsets.set_mode <= action_id < set_mode_end:
        idx = action_id - spec.offsets.set_mode
        mode_count = len(spec.target_modes)
        if mode_count <= 0:
            raise ValueError("No target modes available for SET_MODE actions")
        tower_id = idx // mode_count
        mode = idx % mode_count
        return SetMode(tower_id=tower_id, mode=mode)

    raise ValueError(f"Invalid action_id={action_id}")


def action_to_dict(action: Action) -> dict:
    if isinstance(action, Noop):
        return {"type": ActionType.NOOP.value}
    if isinstance(action, StartWave):
        return {"type": ActionType.START_WAVE.value}
    if isinstance(action, Place):
        return {
            "type": ActionType.PLACE.value,
            "tower_type": int(action.tower_type),
            "cell": int(action.cell),
        }
    if isinstance(action, Upgrade):
        return {"type": ActionType.UPGRADE.value, "tower_id": int(action.tower_id)}
    if isinstance(action, Sell):
        return {"type": ActionType.SELL.value, "tower_id": int(action.tower_id)}
    if isinstance(action, SetMode):
        return {
            "type": ActionType.SET_MODE.value,
            "tower_id": int(action.tower_id),
            "mode": int(action.mode),
        }
    raise TypeError(f"Unsupported action {action!r}")


def action_from_dict(data: dict) -> Action:
    raw_type = str(data.get("type", "")).upper()
    if raw_type == ActionType.NOOP.value:
        return Noop()
    if raw_type == ActionType.START_WAVE.value:
        return StartWave()
    if raw_type == ActionType.PLACE.value:
        return Place(
            tower_type=int(data.get("tower_type", 0)),
            cell=int(data.get("cell", 0)),
        )
    if raw_type == ActionType.UPGRADE.value:
        return Upgrade(tower_id=int(data.get("tower_id", 0)))
    if raw_type == ActionType.SELL.value:
        return Sell(tower_id=int(data.get("tower_id", 0)))
    if raw_type == ActionType.SET_MODE.value:
        return SetMode(
            tower_id=int(data.get("tower_id", 0)),
            mode=int(data.get("mode", 0)),
        )
    raise ValueError(f"Unknown action type {raw_type!r}")
