from pathlib import Path

from vectortd.ai.action_space.discrete_k import build_discrete_k_spec
from vectortd.ai.masking import compute_action_mask_discrete_k
from vectortd.core.model.map import load_map_json
from vectortd.core.model.state import GameState
from vectortd.core.model.towers import BUFF_TOWER_KINDS, get_tower_def, list_tower_defs
from vectortd.core.rules.placement import buildable_cells


TARGET_MODES = ("closest", "weakest", "hardest", "fastest", "random")


def _build_spec():
    root = Path(__file__).resolve().parents[3]
    map_path = root / "data" / "maps" / "switchback.json"
    map_data = load_map_json(map_path)
    cells = buildable_cells(map_data)
    assert cells, "expected at least one buildable cell"

    tower_defs = list(list_tower_defs())
    for kind in BUFF_TOWER_KINDS:
        tower_defs.append(get_tower_def(kind))

    tower_kinds = [tower.kind for tower in tower_defs]
    tower_costs = [int(getattr(tower, "cost", 0)) for tower in tower_defs]
    kind_to_mode_mask = {
        str(tower.kind): tuple(mode in tower.target_modes for mode in TARGET_MODES)
        for tower in tower_defs
    }

    kcells_by_type = [1] * len(tower_defs)
    cells_by_type = [[cells[0]] for _ in tower_defs]

    spec = build_discrete_k_spec(
        max_towers=32,
        map_name=str(getattr(map_data, "name", "")),
        tower_kinds=tower_kinds,
        tower_costs=tower_costs,
        target_modes=TARGET_MODES,
        kind_to_mode_mask=kind_to_mode_mask,
        kcells_by_type=kcells_by_type,
        cells_by_type=cells_by_type,
    )
    return map_data, tower_defs, spec


def test_discrete_k_mask_has_place_actions() -> None:
    map_data, _, spec = _build_spec()
    state = GameState()
    state.bank = 100000
    state.ups = 5
    mask = compute_action_mask_discrete_k(state, None, map_data, spec, phase="BUILD")
    mask_list = list(mask)

    assert len(mask_list) == spec.num_actions
    assert mask_list[spec.offsets.noop]

    place_slice = mask_list[spec.offsets.place : spec.offsets.place + spec.place_count]
    assert any(place_slice)

    upgrade_slice = mask_list[spec.offsets.upgrade : spec.offsets.upgrade + spec.upgrade_count]
    sell_slice = mask_list[spec.offsets.sell : spec.offsets.sell + spec.sell_count]
    set_mode_slice = mask_list[spec.offsets.set_mode : spec.offsets.set_mode + spec.set_mode_count]
    assert not any(upgrade_slice)
    assert not any(sell_slice)
    assert not any(set_mode_slice)


def test_discrete_k_mask_blocks_buff_without_ups() -> None:
    map_data, tower_defs, spec = _build_spec()
    state = GameState()
    state.bank = 100000
    state.ups = 0
    mask = compute_action_mask_discrete_k(state, None, map_data, spec, phase="BUILD")
    mask_list = list(mask)

    for idx, tower in enumerate(tower_defs):
        if tower.kind in BUFF_TOWER_KINDS:
            place_idx = spec.offsets.place + idx
            assert not mask_list[place_idx]


def test_discrete_k_mask_blocks_bank_towers_when_broke() -> None:
    map_data, tower_defs, spec = _build_spec()
    state = GameState()
    state.bank = 0
    state.ups = 5
    mask = compute_action_mask_discrete_k(state, None, map_data, spec, phase="BUILD")
    mask_list = list(mask)

    for idx, tower in enumerate(tower_defs):
        if tower.kind in BUFF_TOWER_KINDS:
            continue
        place_idx = spec.offsets.place + idx
        assert not mask_list[place_idx]
