from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Protocol

from vectortd.ai.actions import (
    Action,
    Noop,
    Place,
    Sell,
    StartWave,
    Upgrade,
    get_tower_slots,
)
from vectortd.ai.env import VectorTDEventEnv
from vectortd.core.model.towers import get_tower_def
from vectortd.core.rules.wave_spawner import LEVELS


logger = logging.getLogger(__name__)


class Policy(Protocol):
    def reset(self, env: VectorTDEventEnv) -> None: ...

    def next_action(self, env: VectorTDEventEnv) -> Action | int: ...


@dataclass
class BuildEntry:
    kind: str
    target_level: int
    label: str
    cell_index: int | None = None


@dataclass
class ReplacementPlan:
    old_label: str
    old_kind: str
    new_kind: str
    target_level: int
    done: bool = False


@dataclass
class HeuristicContext:
    group_indices: list[int]
    positions: dict[int, tuple[int, int]]
    build_sequence: list[BuildEntry]
    replacement_sequence: list[ReplacementPlan]
    replacement_mode: bool = False


def _grid_indices(map_data, cell_x: int, cell_y: int) -> tuple[int, int]:
    grid = int(getattr(map_data, "grid", 25))
    return int(cell_x // grid), int(cell_y // grid)


def _neighbor_cells(ix: int, iy: int) -> list[tuple[int, int]]:
    return [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]


def _border_ok(map_data, ix: int, iy: int) -> bool:
    grid = int(getattr(map_data, "grid", 25))
    cols = max(0, int(getattr(map_data, "width", 0)) // grid)
    rows = max(0, int(getattr(map_data, "height", 0)) // grid)
    return ix >= 3 and iy >= 3 and ix <= (cols - 4) and iy <= (rows - 4)


def select_focus_group(env: VectorTDEventEnv) -> list[int]:
    if env.action_spec is None or env.map_data is None:
        raise RuntimeError("Environment not reset")
    spec = env.action_spec
    map_data = env.map_data
    grid = int(getattr(map_data, "grid", 25))

    cell_positions = list(spec.cell_positions)
    idx_by_pos = {(int(x), int(y)): idx for idx, (x, y) in enumerate(cell_positions)}
    buildable_idx: set[tuple[int, int]] = set()
    for x, y in cell_positions:
        buildable_idx.add(_grid_indices(map_data, x, y))

    path_adjacent: set[tuple[int, int]] = set()
    for ix, iy in buildable_idx:
        if not _border_ok(map_data, ix, iy):
            continue
        for nx, ny in _neighbor_cells(ix, iy):
            if (nx, ny) not in buildable_idx:
                path_adjacent.add((ix, iy))
                break

    path_near: set[tuple[int, int]] = set(path_adjacent)
    for ix, iy in buildable_idx:
        if (ix, iy) in path_adjacent:
            continue
        for nx, ny in _neighbor_cells(ix, iy):
            if (nx, ny) in path_adjacent:
                path_near.add((ix, iy))
                break

    candidates = [cell for cell in sorted(path_near, key=lambda c: (c[1], c[0])) if _border_ok(map_data, *cell)]
    candidate_set = set(candidates)

    for ix, iy in candidates:
        block = [(ix + dx, iy + dy) for dy in range(3) for dx in range(3)]
        if all(cell in candidate_set for cell in block):
            group_indices: list[int] = []
            for cx, cy in block:
                pos = (cx * grid, cy * grid)
                idx = idx_by_pos.get(pos)
                if idx is not None:
                    group_indices.append(idx)
            if len(group_indices) == 9:
                logger.info("focus group found (3x3) at (%s,%s)", ix, iy)
                return group_indices

    for ix, iy in candidates:
        block = [cell for cell in candidate_set if ix <= cell[0] <= ix + 5 and iy <= cell[1] <= iy + 5]
        if len(block) >= 9:
            block_sorted = sorted(block, key=lambda c: (c[1], c[0]))[:9]
            group_indices: list[int] = []
            for cx, cy in block_sorted:
                pos = (cx * grid, cy * grid)
                idx = idx_by_pos.get(pos)
                if idx is not None:
                    group_indices.append(idx)
            if len(group_indices) == 9:
                logger.info("focus group found (5x5 window) at (%s,%s)", ix, iy)
                return group_indices

    if len(candidates) < 9:
        logger.error("Could not find 9 focus cells; found only %s candidates", len(candidates))
        group_cells = candidates
    else:
        group_cells = candidates[:9]
    group_indices: list[int] = []
    for cx, cy in group_cells:
        pos = (cx * grid, cy * grid)
        idx = idx_by_pos.get(pos)
        if idx is not None:
            group_indices.append(idx)
    return group_indices


def init_heuristic_context(env: VectorTDEventEnv, *, verbose: bool) -> HeuristicContext:
    group_indices = select_focus_group(env)
    if env.action_spec is None:
        raise RuntimeError("Missing action spec")
    if len(group_indices) != 9:
        logger.warning("Focus group size=%s (expected 9)", len(group_indices))
    positions = {idx: env.action_spec.cell_positions[idx] for idx in group_indices}
    if verbose:
        logger.info("focus group cells=%s", [positions[idx] for idx in group_indices])
    return HeuristicContext(
        group_indices=group_indices,
        positions=positions,
        build_sequence=_build_sequence(),
        replacement_sequence=_replacement_sequence(),
    )


def _random_action(env: VectorTDEventEnv, rng: random.Random) -> int:
    mask = env.get_action_mask()
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    valid = [idx for idx, ok in enumerate(mask) if ok]
    if not valid:
        return 0
    return rng.choice(valid)


def _tower_slot_by_cell(state, max_towers: int, cell_x: int, cell_y: int) -> tuple[int | None, object | None]:
    tower_slots = get_tower_slots(state, max_towers)
    for idx, tower in enumerate(tower_slots):
        if tower is None:
            continue
        if int(getattr(tower, "cell_x", -1)) == cell_x and int(getattr(tower, "cell_y", -1)) == cell_y:
            return idx, tower
    return None, None


def _tower_at_cell(state, cell_x: int, cell_y: int):
    for tower in getattr(state, "towers", []) or []:
        if int(getattr(tower, "cell_x", -1)) == cell_x and int(getattr(tower, "cell_y", -1)) == cell_y:
            return tower
    return None


def _free_group_cell(state, group_indices: list[int], positions: dict[int, tuple[int, int]]) -> int | None:
    for idx in group_indices:
        pos = positions[idx]
        if _tower_at_cell(state, pos[0], pos[1]) is None:
            return idx
    return None


def _can_afford_place(state, kind: str) -> bool:
    bank = int(getattr(state, "bank", 0))
    return bank >= int(get_tower_def(kind).cost)


def _can_afford_upgrade(state, tower) -> bool:
    if tower is None:
        return False
    if int(getattr(tower, "level", 0)) >= 10:
        return False
    bank = int(getattr(state, "bank", 0))
    upgrade_cost = int(getattr(tower, "base_cost", 0) / 2)
    return bank >= upgrade_cost


def _cheapest_upgrade_action(env: VectorTDEventEnv) -> Upgrade | None:
    if env.engine is None:
        return None
    state = env.engine.state
    bank = int(getattr(state, "bank", 0))
    best_slot = None
    best_cost = None
    tower_slots = get_tower_slots(state, env.max_towers)
    for slot_idx, tower in enumerate(tower_slots):
        if tower is None:
            continue
        if int(getattr(tower, "level", 0)) >= 10:
            continue
        upgrade_cost = int(getattr(tower, "base_cost", 0) / 2)
        if upgrade_cost <= 0 or upgrade_cost > bank:
            continue
        if best_cost is None or upgrade_cost < best_cost:
            best_cost = upgrade_cost
            best_slot = slot_idx
    if best_slot is None:
        return None
    return Upgrade(tower_id=best_slot)


def _build_sequence() -> list[BuildEntry]:
    return [
        BuildEntry(kind="green", target_level=4, label="gl1"),
        BuildEntry(kind="red_refractor", target_level=4, label="red_refractor"),
        BuildEntry(kind="purple1", target_level=1, label="purple1"),
        BuildEntry(kind="blue1", target_level=1, label="blue1_a"),
        BuildEntry(kind="green2", target_level=3, label="green2"),
        BuildEntry(kind="red_spammer", target_level=3, label="red_spammer"),
        BuildEntry(kind="purple2", target_level=1, label="purple2"),
        BuildEntry(kind="blue1", target_level=1, label="blue1_b"),
        BuildEntry(kind="blue2", target_level=1, label="blue2"),
    ]


def _replacement_sequence() -> list[ReplacementPlan]:
    return [
        ReplacementPlan(old_label="gl1", old_kind="green", new_kind="green3", target_level=4),
        ReplacementPlan(old_label="purple1", old_kind="purple1", new_kind="purple3", target_level=4),
    ]


def _base_build_complete(
    env: VectorTDEventEnv,
    build_sequence: list[BuildEntry],
    positions: dict[int, tuple[int, int]],
) -> bool:
    if env.engine is None:
        return False
    state = env.engine.state
    for entry in build_sequence:
        if entry.cell_index is None:
            return False
        pos = positions.get(entry.cell_index)
        if pos is None:
            return False
        tower = _tower_at_cell(state, pos[0], pos[1])
        if tower is None:
            return False
        if int(getattr(tower, "level", 0)) < entry.target_level:
            return False
    return True


def _start_wave_action(env: VectorTDEventEnv) -> Action | None:
    if env.engine is None:
        return None
    state = env.engine.state
    if getattr(state, "game_over", False):
        return None
    if getattr(state, "paused", False):
        return None
    if getattr(state, "creeps", []):
        return None
    if int(getattr(state, "level", 0)) >= len(LEVELS):
        return None
    return StartWave()


def _heuristic_next_action(
    env: VectorTDEventEnv,
    group_indices: list[int],
    positions: dict[int, tuple[int, int]],
    build_sequence: list[BuildEntry],
    replacement_sequence: list[ReplacementPlan],
    *,
    replacement_mode: bool,
) -> Action | None:
    if env.engine is None or env.action_spec is None:
        return None
    spec = env.action_spec
    state = env.engine.state
    bank = int(getattr(state, "bank", 0))
    kind_to_idx = {kind: idx for idx, kind in enumerate(spec.tower_kinds)}

    if not replacement_mode:
        for entry in build_sequence:
            if entry.cell_index is None:
                free_cell = _free_group_cell(state, group_indices, positions)
                if free_cell is None:
                    logger.error("No free cell in focus group for %s", entry.label)
                    return None
                if len(getattr(state, "towers", []) or []) >= env.max_towers:
                    return None
                if not _can_afford_place(state, entry.kind):
                    return None
                action = Place(tower_type=kind_to_idx[entry.kind], cell=free_cell)
                pos = positions[free_cell]
                logger.info("plan: place %s at (%s,%s)", entry.kind, pos[0], pos[1])
                entry.cell_index = free_cell
                return action

            cell_pos = positions[entry.cell_index]
            slot_idx, tower = _tower_slot_by_cell(state, env.max_towers, cell_pos[0], cell_pos[1])
            if tower is None:
                entry.cell_index = None
                return None
            if int(getattr(tower, "level", 0)) < entry.target_level:
                if slot_idx is None:
                    return None
                if not _can_afford_upgrade(state, tower):
                    return None
                action = Upgrade(tower_id=slot_idx)
                logger.info("plan: upgrade %s -> level %s", entry.label, int(getattr(tower, "level", 0)) + 1)
                return action

    if replacement_mode:
        for plan in replacement_sequence:
            if plan.done:
                continue
            entry = next((b for b in build_sequence if b.label == plan.old_label), None)
            if entry is None or entry.cell_index is None:
                plan.done = True
                continue
            cell_pos = positions[entry.cell_index]
            slot_idx, tower = _tower_slot_by_cell(state, env.max_towers, cell_pos[0], cell_pos[1])
            if tower is None:
                if len(getattr(state, "towers", []) or []) >= env.max_towers:
                    return None
                if not _can_afford_place(state, plan.new_kind):
                    return None
                action = Place(tower_type=kind_to_idx[plan.new_kind], cell=entry.cell_index)
                logger.info("replace: place %s at (%s,%s)", plan.new_kind, cell_pos[0], cell_pos[1])
                return action
            tower_kind = str(getattr(tower, "kind", ""))
            if tower_kind == plan.old_kind:
                sale_price = int(getattr(tower, "cost", 0) * 0.75)
                new_cost = int(get_tower_def(plan.new_kind).cost)
                needed = new_cost - sale_price
                if bank >= needed and slot_idx is not None:
                    action = Sell(tower_id=slot_idx)
                    logger.info("replace: sell %s for %s (need %s)", plan.old_label, sale_price, needed)
                    return action
                return None
            if tower_kind == plan.new_kind:
                if int(getattr(tower, "level", 0)) < plan.target_level and slot_idx is not None:
                    if not _can_afford_upgrade(state, tower):
                        return None
                    action = Upgrade(tower_id=slot_idx)
                    logger.info(
                        "replace: upgrade %s -> level %s",
                        plan.new_kind,
                        int(getattr(tower, "level", 0)) + 1,
                    )
                    return action
                plan.done = True
                continue
            logger.error("Unexpected tower kind %s in replacement cell", tower_kind)
            plan.done = True

    upgrade_action = _cheapest_upgrade_action(env)
    if upgrade_action:
        tower_slots = get_tower_slots(state, env.max_towers)
        tower = tower_slots[upgrade_action.tower_id]
        if tower is not None:
            logger.info(
                "upgrade cheapest: %s -> level %s",
                getattr(tower, "kind", "?"),
                int(getattr(tower, "level", 0)) + 1,
            )
        return upgrade_action

    return None


class HeuristicPolicy:
    def __init__(self, *, verbose: bool = False) -> None:
        self._ctx: HeuristicContext | None = None
        self._verbose = verbose

    def reset(self, env: VectorTDEventEnv) -> None:
        self._ctx = init_heuristic_context(env, verbose=self._verbose)

    def next_action(self, env: VectorTDEventEnv) -> Action | int:
        if self._ctx is None:
            self.reset(env)
        if self._ctx is None:
            raise RuntimeError("Heuristic policy not initialized")
        if not self._ctx.replacement_mode and _base_build_complete(
            env,
            self._ctx.build_sequence,
            self._ctx.positions,
        ):
            self._ctx.replacement_mode = True
            logger.info("mode: replacement")
        action = _heuristic_next_action(
            env,
            self._ctx.group_indices,
            self._ctx.positions,
            self._ctx.build_sequence,
            self._ctx.replacement_sequence,
            replacement_mode=self._ctx.replacement_mode,
        )
        if action is None:
            action = _start_wave_action(env) or Noop()
        return action


class RandomPolicy:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, env: VectorTDEventEnv) -> None:
        return None

    def next_action(self, env: VectorTDEventEnv) -> Action | int:
        return _random_action(env, self._rng)


def make_policy(name: str, *, seed: int | None = None, verbose: bool = False) -> Policy:
    if name == "heuristic":
        return HeuristicPolicy(verbose=verbose)
    if name == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unknown policy {name!r}")
