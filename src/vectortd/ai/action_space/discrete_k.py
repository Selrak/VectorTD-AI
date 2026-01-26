from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from vectortd.ai.actions import ActionOffsets
from vectortd.core.rules.placement import buildable_cells


OP_NOOP = "NOOP"
OP_START_WAVE = "START_WAVE"
OP_PLACE = "PLACE"
OP_UPGRADE = "UPGRADE"
OP_SELL = "SELL"
OP_SET_MODE = "SET_MODE"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    op: str
    t: int | None
    k: int | None
    slot: int | None
    mode: int | None


@dataclass(frozen=True, slots=True)
class DiscreteKSpec:
    max_towers: int
    map_name: str
    tower_kinds: tuple[str, ...]
    tower_costs: tuple[int, ...]
    target_modes: tuple[str, ...]
    kind_to_mode_mask: dict[str, tuple[bool, ...]]
    kcells_by_type: tuple[int, ...]
    cells_by_type: tuple[tuple[tuple[int, int] | None, ...], ...]
    offsets: ActionOffsets
    place_count: int
    upgrade_count: int
    sell_count: int
    set_mode_count: int
    num_actions: int


def _load_gym():
    try:
        import gymnasium as gym

        return gym
    except Exception:
        pass
    try:
        import gym

        return gym
    except Exception as exc:  # pragma: no cover - handled in tests via importorskip
        raise ModuleNotFoundError(
            "gymnasium or gym is required for DiscreteKActionTable"
        ) from exc


def _normalize_kcells_by_type(
    tower_types: Sequence[str],
    kcells_by_type: Mapping[str, int] | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(kcells_by_type, Mapping):
        missing = [tower_type for tower_type in tower_types if tower_type not in kcells_by_type]
        if missing:
            raise ValueError(f"Missing Kcells for tower types: {', '.join(missing)}")
        kcells = tuple(int(kcells_by_type[tower_type]) for tower_type in tower_types)
    else:
        kcells = tuple(int(value) for value in kcells_by_type)
        if len(kcells) != len(tower_types):
            raise ValueError("Kcells length must match tower_types length")
    if any(value < 0 for value in kcells):
        raise ValueError("Kcells values must be >= 0")
    return kcells


def build_discrete_k_spec(
    *,
    max_towers: int,
    map_name: str,
    tower_kinds: Sequence[str],
    tower_costs: Sequence[int],
    target_modes: Sequence[str],
    kind_to_mode_mask: dict[str, tuple[bool, ...]],
    kcells_by_type: Sequence[int],
    cells_by_type: Sequence[Sequence[tuple[int, int] | None]],
) -> DiscreteKSpec:
    kcells = tuple(int(value) for value in kcells_by_type)
    place_count = sum(kcells)
    upgrade_count = int(max_towers)
    sell_count = int(max_towers)
    mode_count = len(target_modes)
    set_mode_count = int(max_towers) * mode_count
    offsets = ActionOffsets(
        noop=0,
        start_wave=1,
        place=2,
        upgrade=2 + place_count,
        sell=2 + place_count + upgrade_count,
        set_mode=2 + place_count + upgrade_count + sell_count,
    )
    num_actions = offsets.set_mode + set_mode_count
    return DiscreteKSpec(
        max_towers=int(max_towers),
        map_name=str(map_name),
        tower_kinds=tuple(str(kind) for kind in tower_kinds),
        tower_costs=tuple(int(cost) for cost in tower_costs),
        target_modes=tuple(str(mode) for mode in target_modes),
        kind_to_mode_mask=dict(kind_to_mode_mask),
        kcells_by_type=kcells,
        cells_by_type=tuple(tuple(cells) for cells in cells_by_type),
        offsets=offsets,
        place_count=place_count,
        upgrade_count=upgrade_count,
        sell_count=sell_count,
        set_mode_count=set_mode_count,
        num_actions=num_actions,
    )


class DiscreteKActionTable:
    def __init__(
        self,
        tower_types: Sequence[str],
        kcells_by_type: Mapping[str, int] | Sequence[int],
        ktower: int,
        modes: Sequence[str],
    ) -> None:
        self.tower_types = tuple(str(tower_type) for tower_type in tower_types)
        self.modes = tuple(str(mode) for mode in modes)
        self.ktower = int(ktower)
        if self.ktower < 0:
            raise ValueError("Ktower must be >= 0")
        self.kcells_by_type = _normalize_kcells_by_type(self.tower_types, kcells_by_type)

    def build_for_map(
        self,
        map_data,
        *,
        cells_by_type: Mapping[int, Sequence[tuple[int, int] | None]] | None = None,
    ):
        if cells_by_type is None:
            base_cells = list(buildable_cells(map_data))
            base_cells.sort(key=lambda cell: (cell[1], cell[0]))
            cells_by_type = {}
            for idx, kcells in enumerate(self.kcells_by_type):
                selected = list(base_cells[:kcells])
                if len(selected) < kcells:
                    selected.extend([None] * (kcells - len(selected)))
                cells_by_type[idx] = selected
        normalized_cells: dict[int, list[tuple[int, int] | None]] = {}
        for idx, kcells in enumerate(self.kcells_by_type):
            raw = list(cells_by_type.get(idx, [])) if cells_by_type is not None else []
            if len(raw) < kcells:
                raw.extend([None] * (kcells - len(raw)))
            elif len(raw) > kcells:
                raw = raw[:kcells]
            normalized_cells[idx] = raw

        table: list[ActionSpec] = [
            ActionSpec(op=OP_NOOP, t=None, k=None, slot=None, mode=None),
            ActionSpec(op=OP_START_WAVE, t=None, k=None, slot=None, mode=None),
        ]
        for t_idx, kcells in enumerate(self.kcells_by_type):
            for k in range(kcells):
                table.append(ActionSpec(op=OP_PLACE, t=t_idx, k=k, slot=None, mode=None))
        for slot in range(self.ktower):
            table.append(ActionSpec(op=OP_UPGRADE, t=None, k=None, slot=slot, mode=None))
        for slot in range(self.ktower):
            table.append(ActionSpec(op=OP_SELL, t=None, k=None, slot=slot, mode=None))
        for slot in range(self.ktower):
            for mode_idx in range(len(self.modes)):
                table.append(
                    ActionSpec(op=OP_SET_MODE, t=None, k=None, slot=slot, mode=mode_idx)
                )

        gym = _load_gym()
        action_space = gym.spaces.Discrete(len(table))
        return action_space, table, normalized_cells
