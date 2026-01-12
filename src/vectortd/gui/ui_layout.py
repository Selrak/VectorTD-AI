from __future__ import annotations

from dataclasses import dataclass

import pyglet


@dataclass
class GridLayout:
    origin_x: float
    origin_y_top: float
    columns: int
    cell_size: float
    spacing: float
    _index: int = 0

    def add_cell(self) -> tuple[float, float, float, float]:
        row = self._index // self.columns
        col = self._index % self.columns
        x = self.origin_x + col * (self.cell_size + self.spacing)
        y = self.origin_y_top - self.cell_size - row * (self.cell_size + self.spacing)
        self._index += 1
        return x, y, self.cell_size, self.cell_size


class SidebarLayout:
    def __init__(
        self,
        *,
        x: float,
        y_top: float,
        width: float,
        padding: float = 12,
        spacing: float = 10,
        min_y: float | None = None,
    ) -> None:
        self.x = x + padding
        self.y_top = y_top - padding
        self.width = max(0.0, width - 2 * padding)
        self.spacing = spacing
        self._cursor = self.y_top
        self._min_y = min_y

    def add_label(
        self,
        text: str,
        *,
        font_size: int,
        color: tuple[int, int, int, int],
        batch: pyglet.graphics.Batch,
        anchor_x: str = "left",
    ) -> pyglet.text.Label:
        label = pyglet.text.Label(
            text,
            x=self.x,
            y=self._cursor,
            anchor_x=anchor_x,
            anchor_y="top",
            font_size=font_size,
            color=color,
            batch=batch,
        )
        height = max(label.content_height, float(font_size))
        self._cursor -= height + self.spacing
        if self._min_y is not None:
            self._cursor = max(self._cursor, self._min_y)
        return label

    def add_spacer(self, height: float) -> None:
        self._cursor -= max(0.0, height)
        if self._min_y is not None:
            self._cursor = max(self._cursor, self._min_y)

    def add_grid(
        self,
        *,
        rows: int,
        columns: int,
        cell_size: float,
        spacing: float = 8,
    ) -> GridLayout:
        grid_height = rows * cell_size + max(0, rows - 1) * spacing
        grid = GridLayout(
            origin_x=self.x,
            origin_y_top=self._cursor,
            columns=max(1, columns),
            cell_size=cell_size,
            spacing=spacing,
        )
        self._cursor -= grid_height + self.spacing
        return grid

    def add_box(self, height: float) -> tuple[float, float, float, float]:
        box_height = max(0.0, height)
        if self._min_y is not None:
            box_height = min(box_height, max(0.0, self._cursor - self._min_y))
        y = self._cursor - box_height
        bounds = (self.x, y, self.width, box_height)
        self._cursor -= box_height + self.spacing
        if self._min_y is not None:
            self._cursor = max(self._cursor, self._min_y)
        return bounds

    def remaining_height(self) -> float:
        if self._min_y is None:
            return self._cursor
        return max(0.0, self._cursor - self._min_y)


class BottomLayout:
    def __init__(
        self,
        *,
        x: float,
        y_bottom: float,
        width: float,
        padding: float = 12,
        spacing: float = 10,
    ) -> None:
        self.x = x + padding
        self.y_bottom = y_bottom + padding
        self.width = max(0.0, width - 2 * padding)
        self.spacing = spacing
        self._cursor = self.y_bottom

    @property
    def cursor_y(self) -> float:
        return self._cursor

    def add_box(self, height: float) -> tuple[float, float, float, float]:
        box_height = max(0.0, height)
        bounds = (self.x, self._cursor, self.width, box_height)
        self._cursor += box_height + self.spacing
        return bounds
