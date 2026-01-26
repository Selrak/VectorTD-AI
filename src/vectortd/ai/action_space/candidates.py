from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


DIST_PENALTY_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class ScoredCell:
    cell: tuple[int, int]
    center: tuple[float, float]
    score: float
    coverage: float
    dist_to_path: float


def compute_path_samples(map_data, step_px: float = 10.0) -> list[tuple[float, float]]:
    if step_px <= 0:
        raise ValueError("step_px must be > 0")

    def marker_xy(marker_id: int) -> tuple[float, float]:
        marker_fn = getattr(map_data, "marker_xy", None)
        if callable(marker_fn):
            return marker_fn(marker_id)
        markers = getattr(map_data, "markers", {}) or {}
        if marker_id not in markers:
            raise KeyError(marker_id)
        x, y = markers[marker_id]
        if x is None or y is None:
            raise ValueError(marker_id)
        return float(x), float(y)

    samples: list[tuple[float, float]] = []
    for path in getattr(map_data, "paths", []) or []:
        if not path or len(path) < 2:
            continue
        for a, b in zip(path[:-1], path[1:]):
            try:
                x1, y1 = marker_xy(int(a))
                x2, y2 = marker_xy(int(b))
            except Exception:
                continue
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0.0:
                continue
            steps = max(1, int(math.ceil(length / float(step_px))))
            for i in range(steps + 1):
                t = i / steps
                x = x1 + dx * t
                y = y1 + dy * t
                if samples:
                    last_x, last_y = samples[-1]
                    if abs(last_x - x) < 1e-6 and abs(last_y - y) < 1e-6:
                        continue
                samples.append((float(x), float(y)))
    return samples


def score_cells_for_type(
    buildable_cells: Sequence[tuple[int, int]],
    path_samples: Sequence[tuple[float, float]],
    range_px: float,
    *,
    grid: int = 25,
    dist_penalty_weight: float = DIST_PENALTY_WEIGHT,
) -> list[ScoredCell]:
    if range_px <= 0:
        raise ValueError("range_px must be > 0")
    if grid <= 0:
        raise ValueError("grid must be > 0")

    total_samples = len(path_samples)
    range_px = float(range_px)
    range_sq = range_px * range_px
    scored: list[ScoredCell] = []
    for cell_x, cell_y in buildable_cells:
        cx = float(cell_x) + grid * 0.5
        cy = float(cell_y) + grid * 0.5
        if total_samples:
            in_range = 0
            min_dist_sq = float("inf")
            for px, py in path_samples:
                dx = px - cx
                dy = py - cy
                dist_sq = dx * dx + dy * dy
                if dist_sq <= range_sq:
                    in_range += 1
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
            coverage = in_range / total_samples
            dist_to_path = math.sqrt(min_dist_sq) if min_dist_sq != float("inf") else range_px
        else:
            coverage = 0.0
            dist_to_path = range_px

        dist_norm = dist_to_path / range_px
        if dist_norm < 0.0:
            dist_norm = 0.0
        elif dist_norm > 1.0:
            dist_norm = 1.0
        score = float(coverage) - dist_penalty_weight * dist_norm
        scored.append(
            ScoredCell(
                cell=(int(cell_x), int(cell_y)),
                center=(cx, cy),
                score=score,
                coverage=float(coverage),
                dist_to_path=float(dist_to_path),
            )
        )
    return scored


def select_top_k_diverse(
    scored: Sequence[ScoredCell],
    k: int,
    dmin_px: float,
) -> list[tuple[int, int]]:
    if k <= 0:
        return []
    dmin_px = float(dmin_px)
    sorted_scored = sorted(scored, key=lambda item: item.score, reverse=True)
    if dmin_px <= 0.0:
        return [item.cell for item in sorted_scored[:k]]

    selected: list[tuple[int, int]] = []
    selected_centers: list[tuple[float, float]] = []
    for item in sorted_scored:
        if len(selected) >= k:
            break
        cx, cy = item.center
        if any(max(abs(cx - sx), abs(cy - sy)) < dmin_px for sx, sy in selected_centers):
            continue
        selected.append(item.cell)
        selected_centers.append(item.center)
    return selected
