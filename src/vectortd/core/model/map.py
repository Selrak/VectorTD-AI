from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal


@dataclass(slots=True)
class MapData:
    name: str
    level_index: int
    width: int
    height: int
    grid: int
    spawn_dir: Literal["up", "down", "left", "right"]
    paths: list[list[int]]
    markers: dict[int, tuple[float | None, float | None]]
    buildable_grid_mode: str | None = None
    buildable_cells: list[tuple[int, int]] | None = None

    def marker_xy(self, marker_id: int) -> tuple[float, float]:
        x, y = self.markers[marker_id]
        if x is None or y is None:
            raise ValueError(f"Marker {marker_id} has no coordinates")
        return x, y


def load_map_json(path: str | Path) -> MapData:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    world = data.get("world", {})
    markers_raw = data["markers"]
    markers: dict[int, tuple[float | None, float | None]] = {}
    for k, v in markers_raw.items():
        x, y = v[0], v[1]
        if x is None or y is None:
            markers[int(k)] = (None, None)
        else:
            markers[int(k)] = (float(x), float(y))

    buildable = data.get("buildable", {}) or {}
    allowed_cells = buildable.get("allowed_cells")
    parsed_cells = None
    if isinstance(allowed_cells, list):
        parsed_cells = []
        for cell in allowed_cells:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                continue
            parsed_cells.append((int(cell[0]), int(cell[1])))

    return MapData(
        name=str(data["name"]),
        level_index=int(data.get("level_index", 0)),
        width=int(world.get("width", 550)),
        height=int(world.get("height", 450)),
        grid=int(world.get("grid", 25)),
        spawn_dir=data["spawn_dir"],
        paths=[[int(x) for x in path] for path in data["paths"]],
        markers=markers,
        buildable_grid_mode=buildable.get("grid_mode"),
        buildable_cells=parsed_cells,
    )
