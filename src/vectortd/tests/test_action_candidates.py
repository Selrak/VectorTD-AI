from pathlib import Path

from vectortd.ai.action_space.candidates import (
    ScoredCell,
    compute_path_samples,
    score_cells_for_type,
    select_top_k_diverse,
)
from vectortd.core.model.map import load_map_json
from vectortd.core.rules.placement import buildable_cells


def test_compute_path_samples_switchback_is_non_empty() -> None:
    root = Path(__file__).resolve().parents[3]
    map_path = root / "data" / "maps" / "switchback.json"
    map_data = load_map_json(map_path)
    samples = compute_path_samples(map_data, step_px=10.0)
    assert len(samples) > 0


def test_select_top_k_diverse_respects_min_distance() -> None:
    scored = [
        ScoredCell(cell=(0, 0), center=(0.0, 0.0), score=1.0, coverage=0.0, dist_to_path=0.0),
        ScoredCell(cell=(1, 1), center=(40.0, 40.0), score=0.9, coverage=0.0, dist_to_path=0.0),
        ScoredCell(cell=(2, 0), center=(100.0, 0.0), score=0.8, coverage=0.0, dist_to_path=0.0),
    ]
    selected = select_top_k_diverse(scored, k=2, dmin_px=50.0)
    assert selected == [(0, 0), (2, 0)]


def test_switchback_candidates_return_requested_count() -> None:
    root = Path(__file__).resolve().parents[3]
    map_path = root / "data" / "maps" / "switchback.json"
    map_data = load_map_json(map_path)
    cells = buildable_cells(map_data)
    samples = compute_path_samples(map_data, step_px=10.0)
    scored = score_cells_for_type(cells, samples, range_px=70.0, grid=map_data.grid)
    selected = select_top_k_diverse(scored, k=32, dmin_px=float(map_data.grid))
    assert len(selected) == 32
