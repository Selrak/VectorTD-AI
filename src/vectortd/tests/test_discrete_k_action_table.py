from types import SimpleNamespace

import pytest

try:
    import gymnasium as _gymnasium  # noqa: F401
except Exception:
    pytest.importorskip("gym")

from vectortd.ai.action_space.discrete_k import (  # noqa: E402
    OP_NOOP,
    OP_PLACE,
    OP_SELL,
    OP_SET_MODE,
    OP_START_WAVE,
    OP_UPGRADE,
    ActionSpec,
    DiscreteKActionTable,
)


def _make_map():
    return SimpleNamespace(
        name="unit_test_map",
        buildable_cells=[(50, 0), (0, 0), (25, 25), (0, 25)],
        grid=25,
        width=100,
        height=100,
        paths=[],
        markers={},
    )


def test_discrete_k_action_table_order_and_stability() -> None:
    map_data = _make_map()
    tower_types = ("green", "red_refractor")
    kcells_by_type = {"green": 2, "red_refractor": 3}
    ktower = 4
    modes = ("closest", "fastest")

    table_builder = DiscreteKActionTable(
        tower_types=tower_types,
        kcells_by_type=kcells_by_type,
        ktower=ktower,
        modes=modes,
    )

    action_space, specs, cells_by_type = table_builder.build_for_map(map_data)
    assert action_space.n == len(specs)

    expected_cells = sorted(map_data.buildable_cells, key=lambda cell: (cell[1], cell[0]))
    assert cells_by_type[0] == expected_cells[:2]
    assert cells_by_type[1] == expected_cells[:3]

    place_count = sum(kcells_by_type.values())
    assert specs[0] == ActionSpec(op=OP_NOOP, t=None, k=None, slot=None, mode=None)
    assert specs[1] == ActionSpec(op=OP_START_WAVE, t=None, k=None, slot=None, mode=None)
    place_specs = specs[2 : 2 + place_count]
    assert [(spec.op, spec.t, spec.k) for spec in place_specs] == [
        (OP_PLACE, 0, 0),
        (OP_PLACE, 0, 1),
        (OP_PLACE, 1, 0),
        (OP_PLACE, 1, 1),
        (OP_PLACE, 1, 2),
    ]

    upgrade_start = 2 + place_count
    upgrade_specs = specs[upgrade_start : upgrade_start + ktower]
    assert [(spec.op, spec.slot) for spec in upgrade_specs] == [
        (OP_UPGRADE, 0),
        (OP_UPGRADE, 1),
        (OP_UPGRADE, 2),
        (OP_UPGRADE, 3),
    ]

    sell_start = upgrade_start + ktower
    sell_specs = specs[sell_start : sell_start + ktower]
    assert [(spec.op, spec.slot) for spec in sell_specs] == [
        (OP_SELL, 0),
        (OP_SELL, 1),
        (OP_SELL, 2),
        (OP_SELL, 3),
    ]

    set_mode_start = sell_start + ktower
    set_mode_specs = specs[set_mode_start:]
    assert len(set_mode_specs) == ktower * len(modes)
    assert set_mode_specs[0] == ActionSpec(op=OP_SET_MODE, t=None, k=None, slot=0, mode=0)
    assert set_mode_specs[1] == ActionSpec(op=OP_SET_MODE, t=None, k=None, slot=0, mode=1)
    assert set_mode_specs[2] == ActionSpec(op=OP_SET_MODE, t=None, k=None, slot=1, mode=0)

    action_space2, specs2, cells_by_type2 = table_builder.build_for_map(map_data)
    assert action_space2.n == action_space.n
    assert specs2 == specs
    assert cells_by_type2 == cells_by_type
