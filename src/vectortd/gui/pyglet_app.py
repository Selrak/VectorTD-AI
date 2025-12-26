from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable, List

import pyglet
from pyglet import gl
from pyglet.math import Mat4

from .assets import load_creep_images, load_map_image, load_tower_images
from .ui_layout import SidebarLayout
from ..core.model.map import MapData, load_map_json
from ..core.model.state import GameState
from ..core.model.towers import get_tower_def
from ..core.rules.creep_motion import step_creeps
from ..core.rules.wave_spawner import start_next_wave
from ..core.rules.placement import buildable_cells, can_place_tower, place_tower, upgrade_tower


SPEED_OPTIONS: tuple[tuple[str, float], ...] = (
    ("0.1x", 0.1),
    ("0.25x", 0.25),
    ("0.5x", 0.5),
    ("0.75x", 0.75),
    ("1x", 1.0),
    ("2x", 2.0),
    ("3x", 3.0),
    ("5x", 5.0),
    ("10x", 10.0),
    ("20x", 20.0),
    ("30x", 30.0),
    ("50x", 50.0),
    ("100x", 100.0),
)


@dataclass
class _SamplePoint:
    x: float
    y: float
    dx: float
    dy: float


def _markers_ready(map_data: MapData) -> bool:
    if not map_data.markers:
        return False
    for x, y in map_data.markers.values():
        if x is None or y is None:
            return False
    return True


def _sample_polyline(points: Iterable[tuple[float, float]], count: int) -> List[_SamplePoint]:
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("Polyline needs at least 2 points")
    if count < 2:
        raise ValueError("Need at least 2 samples")

    segments = []
    total_len = 0.0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        segments.append((x1, y1, x2, y2, dx / seg_len, dy / seg_len, seg_len))
        total_len += seg_len

    if not segments:
        raise ValueError("Polyline has no length")

    step = total_len / (count - 1)
    samples: List[_SamplePoint] = []
    seg_idx = 0
    seg_start = 0.0

    for i in range(count):
        target = i * step
        while seg_idx < len(segments) - 1 and (seg_start + segments[seg_idx][6]) < target:
            seg_start += segments[seg_idx][6]
            seg_idx += 1
        x1, y1, x2, y2, dx, dy, seg_len = segments[seg_idx]
        t = 0.0 if seg_len == 0.0 else (target - seg_start) / seg_len
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        samples.append(_SamplePoint(x=x, y=y, dx=dx, dy=dy))

    return samples


def _fallback_markers(map_data: MapData) -> dict[int, tuple[float, float]]:
    width = float(map_data.width)
    height = float(map_data.height)

    margin = min(width, height) * 0.08
    left = margin
    right = width - margin
    top = margin
    bottom = height - margin
    mid1 = top + (bottom - top) * 0.33
    mid2 = top + (bottom - top) * 0.66

    if map_data.spawn_dir == "up":
        centerline = [
            (left, top),
            (right, top),
            (right, mid1),
            (left, mid1),
            (left, mid2),
            (right, mid2),
            (right, bottom),
        ]
    else:
        centerline = [
            (left, height / 2),
            (right, height / 2),
        ]

    lane_offset = 10.0
    markers: dict[int, tuple[float, float]] = {}

    for lane_idx, path in enumerate(map_data.paths):
        samples = _sample_polyline(centerline, len(path))
        sign = 1.0 if lane_idx % 2 == 0 else -1.0
        for marker_id, sample in zip(path, samples):
            nx = -sample.dy
            ny = sample.dx
            mx = sample.x + nx * lane_offset * sign
            my = sample.y + ny * lane_offset * sign
            markers[int(marker_id)] = (mx, my)

    return markers


def _next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            idx = int(child.name)
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    next_idx = max_idx + 1
    run_dir = base_dir / str(next_idx)
    while run_dir.exists():
        next_idx += 1
        run_dir = base_dir / str(next_idx)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class SimpleGui:
    def __init__(self, map_path: Path, run_dir: Path) -> None:
        self.state = GameState()
        self.map_data = load_map_json(map_path)

        self.map_image = load_map_image(self.map_data.name)
        self._world_offset = (
            max(0.0, (self.map_image.width - self.map_data.width) / 2.0),
            max(0.0, (self.map_image.height - self.map_data.height) / 2.0),
        )
        self._sidebar_width = 220
        self._sidebar_padding = 12
        self._sidebar_x = self.map_image.width
        self._logical_width = self.map_image.width + self._sidebar_width
        self._logical_height = self.map_image.height
        self._window_width = self._logical_width
        self._window_height = self._logical_height
        self._view_scale = 1.0
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0

        if not _markers_ready(self.map_data):
            self.map_data.markers = _fallback_markers(self.map_data)

        start_next_wave(self.state, self.map_data)

        self.window = pyglet.window.Window(
            width=self._window_width,
            height=self._window_height,
            caption=f"VectorTD - {self.map_data.name}",
        )
        self._set_window_to_screen()
        self.batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()
        self.map_sprite = pyglet.sprite.Sprite(self.map_image, x=0, y=0, batch=self.batch)
        self._grid_shapes: list[pyglet.shapes.Rectangle] = []
        self._build_grid_overlay()

        self.tower_images = load_tower_images()
        self.tower_sprites: List[pyglet.sprite.Sprite] = []
        self._placement_active = False
        self._placement_kind = "green"
        self._placement_cell: tuple[int, int] | None = None
        self._placement_valid = False
        self._tower_preview_sprite = pyglet.sprite.Sprite(
            self.tower_images[self._placement_kind],
            x=-1000,
            y=-1000,
            batch=self.batch,
        )
        self._tower_preview_sprite.opacity = 0
        self._last_mouse_pos: tuple[float, float] | None = None
        self._selected_tower = None
        self._hovered_tower_kind: str | None = None
        self._tower_button_enabled = True
        self._tower_range_shape = pyglet.shapes.Circle(
            -1000,
            -1000,
            radius=1,
            color=(80, 200, 120),
            batch=self.batch,
        )
        self._tower_range_shape.opacity = 0

        self.creep_images = load_creep_images()
        self.creep_sprites: List[pyglet.sprite.Sprite] = []
        self._sync_creep_sprites()

        self._capture_every = 100
        self._tick_count = 0
        self._pending_capture = False
        self._screenshot_dir = run_dir / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._speed_index = next(
            (i for i, (_, factor) in enumerate(SPEED_OPTIONS) if factor == 1.0),
            0,
        )
        self._speed_factor = SPEED_OPTIONS[self._speed_index][1]

        self._last_lives = self.state.lives
        self._last_bank = self.state.bank
        self._build_sidebar_ui()

        self.window.push_handlers(
            on_draw=self.on_draw,
            on_resize=self.on_resize,
            on_mouse_press=self.on_mouse_press,
            on_mouse_motion=self.on_mouse_motion,
            on_mouse_drag=self.on_mouse_drag,
        )
        self._apply_viewport()
        self._apply_projection()
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _sync_creep_sprites(self) -> None:
        creeps = self.state.creeps
        if len(self.creep_sprites) < len(creeps):
            for c in creeps[len(self.creep_sprites):]:
                img = self.creep_images.get(c.type_id, self.creep_images[1])
                draw_x = c.x + self._world_offset[0]
                draw_y = self.map_image.height - (c.y + self._world_offset[1])
                sprite = pyglet.sprite.Sprite(
                    img,
                    x=draw_x,
                    y=draw_y,
                    batch=self.batch,
                )
                self.creep_sprites.append(sprite)

        for sprite, creep in zip(self.creep_sprites, creeps):
            sprite.x = creep.x + self._world_offset[0]
            sprite.y = self.map_image.height - (creep.y + self._world_offset[1])

    def _sync_tower_sprites(self) -> None:
        towers = self.state.towers
        default_img = self.tower_images.get(self._placement_kind)
        if len(self.tower_sprites) < len(towers):
            for tower in towers[len(self.tower_sprites):]:
                img = self.tower_images.get(tower.kind, default_img)
                if img is None:
                    continue
                draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
                sprite = pyglet.sprite.Sprite(img, x=draw_x, y=draw_y, batch=self.batch)
                self.tower_sprites.append(sprite)

        for sprite, tower in zip(self.tower_sprites, towers):
            draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
            sprite.x = draw_x
            sprite.y = draw_y

    def update(self, dt: float) -> None:
        frame_steps = max(0.0, dt) * 60.0 * self._speed_factor
        whole_steps = int(frame_steps)
        remainder = frame_steps - whole_steps
        for _ in range(whole_steps):
            step_creeps(self.state, self.map_data, dt_scale=1.0)
        if remainder > 0.0:
            step_creeps(self.state, self.map_data, dt_scale=remainder)
        if self.state.lives != self._last_lives:
            self._last_lives = self.state.lives
            self._lives_label.text = f"Lives: {self.state.lives}"
        if self.state.bank != self._last_bank:
            self._last_bank = self.state.bank
            self._money_label.text = f"Money: ${self.state.bank}"
            self._refresh_tower_button()
            self._refresh_upgrade_button()
        if self.state.lives <= 0 and not self.state.game_over:
            self.state.game_over = True
        self._set_game_over_overlay(self.state.game_over)
        self._sync_creep_sprites()
        self._sync_tower_sprites()
        self._tick_count += 1
        if self._tick_count % self._capture_every == 0:
            self._pending_capture = True

    def on_draw(self) -> None:
        self.window.clear()
        self.batch.draw()
        self.ui_batch.draw()
        if self._pending_capture:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            filename = self._screenshot_dir / f"tick_{self._tick_count:06d}.png"
            buffer.save(str(filename))
            self._pending_capture = False

    def _apply_viewport(self) -> None:
        fb_w, fb_h = self.window.get_framebuffer_size()
        gl.glViewport(0, 0, fb_w, fb_h)

    def _apply_projection(self) -> None:
        self.window.projection = Mat4.orthogonal_projection(
            0,
            self._window_width,
            0,
            self._window_height,
            -1,
            1,
        )
        self.window.view = Mat4().translate(
            (self._view_offset_x, self._view_offset_y, 0.0),
        ).scale(
            (self._view_scale, self._view_scale, 1.0),
        )

    def on_resize(self, width: int, height: int) -> None:
        self._window_width = width
        self._window_height = height
        self._update_view_transform()
        self._apply_viewport()
        self._apply_projection()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        logical_x, logical_y = self._to_logical_coords(x, y)
        in_down = _point_in_rect(logical_x, logical_y, self._speed_down_bounds)
        in_up = _point_in_rect(logical_x, logical_y, self._speed_up_bounds)
        in_tower = _point_in_rect(logical_x, logical_y, self._tower_button_bounds)
        in_upgrade = self._upgrade_visible and _point_in_rect(logical_x, logical_y, self._upgrade_button_bounds)
        print(
            "[gui] click "
            f"raw=({x},{y}) logical=({logical_x:.1f},{logical_y:.1f}) "
            f"button={button} down={in_down} up={in_up}"
        )
        if in_upgrade:
            self._handle_upgrade_click()
            return
        if in_tower and self._tower_button_enabled:
            self._set_placement_active(not self._placement_active)
            return
        if in_down:
            self._set_speed_index(self._speed_index - 1)
        elif in_up:
            self._set_speed_index(self._speed_index + 1)
        elif self._placement_active:
            cell = self._grid_cell_at(logical_x, logical_y)
            if cell is not None:
                cell_x, cell_y = cell
                if can_place_tower(self.state, self.map_data, cell_x, cell_y, self._placement_kind):
                    place_tower(self.state, self.map_data, cell_x, cell_y, self._placement_kind)
                    self._sync_tower_sprites()
                    self._set_placement_active(False)
            self._update_placement_preview(logical_x, logical_y)
        else:
            self._handle_tower_selection(logical_x, logical_y)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        logical_x, logical_y = self._to_logical_coords(x, y)
        self._last_mouse_pos = (logical_x, logical_y)
        self._update_placement_preview(logical_x, logical_y)
        self._update_tower_button_hover(logical_x, logical_y)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        logical_x, logical_y = self._to_logical_coords(x, y)
        self._last_mouse_pos = (logical_x, logical_y)
        self._update_placement_preview(logical_x, logical_y)
        self._update_tower_button_hover(logical_x, logical_y)

    def _build_sidebar_ui(self) -> None:
        sidebar_height = self._logical_height
        sidebar_color = (34, 36, 40)
        self._sidebar_bg = pyglet.shapes.Rectangle(
            self._sidebar_x,
            0,
            self._sidebar_width,
            sidebar_height,
            color=sidebar_color,
            batch=self.ui_batch,
        )

        layout = SidebarLayout(
            x=self._sidebar_x,
            y_top=sidebar_height,
            width=self._sidebar_width,
            padding=self._sidebar_padding,
            spacing=12,
        )
        self._lives_label = layout.add_label(
            f"Lives: {self.state.lives}",
            font_size=14,
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._money_label = layout.add_label(
            f"Money: ${self.state.bank}",
            font_size=14,
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )

        self._tower_title_label = layout.add_label(
            "Tower",
            font_size=12,
            color=(200, 200, 200, 255),
            batch=self.ui_batch,
        )
        tower_button_size = 64
        tower_grid = layout.add_grid(
            rows=1,
            columns=2,
            cell_size=tower_button_size,
            spacing=10,
        )
        tower_button_x, tower_button_y, _, _ = tower_grid.add_cell()
        self._tower_button = pyglet.shapes.BorderedRectangle(
            tower_button_x,
            tower_button_y,
            tower_button_size,
            tower_button_size,
            color=(60, 60, 64),
            border=2,
            border_color=(110, 110, 115),
            batch=self.ui_batch,
        )
        tower_icon = self.tower_images[self._placement_kind]
        self._tower_button_sprite = pyglet.sprite.Sprite(
            tower_icon,
            x=tower_button_x + tower_button_size / 2,
            y=tower_button_y + tower_button_size / 2,
            batch=self.ui_batch,
        )
        icon_scale = min(
            (tower_button_size - 12) / tower_icon.width,
            (tower_button_size - 12) / tower_icon.height,
        )
        self._tower_button_sprite.scale = icon_scale
        self._tower_button_bounds = (tower_button_x, tower_button_y, tower_button_size, tower_button_size)
        self._refresh_tower_button()

        self._tower_info_title = layout.add_label(
            "Tower Info",
            font_size=12,
            color=(200, 200, 200, 255),
            batch=self.ui_batch,
        )
        info_x, info_y, info_w, info_h = layout.add_box(170)
        self._tower_info_box = pyglet.shapes.BorderedRectangle(
            info_x,
            info_y,
            info_w,
            info_h,
            color=(26, 28, 32),
            border=2,
            border_color=(70, 72, 76),
            batch=self.ui_batch,
        )
        self._tower_info_bounds = (info_x, info_y, info_w, info_h)
        self._tower_info_label = pyglet.text.Label(
            "",
            x=info_x + 8,
            y=info_y + info_h - 8,
            width=int(info_w - 16),
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            font_size=10,
            color=(220, 220, 220, 255),
            batch=self.ui_batch,
        )

        upgrade_row_x, upgrade_row_y, upgrade_row_w, upgrade_row_h = layout.add_box(36)
        upgrade_button_width = min(120, int(upgrade_row_w * 0.65))
        upgrade_button_height = 26
        upgrade_button_x = upgrade_row_x
        upgrade_button_y = upgrade_row_y + (upgrade_row_h - upgrade_button_height) / 2
        self._upgrade_button = pyglet.shapes.BorderedRectangle(
            upgrade_button_x,
            upgrade_button_y,
            upgrade_button_width,
            upgrade_button_height,
            color=(60, 60, 64),
            border=2,
            border_color=(110, 110, 115),
            batch=self.ui_batch,
        )
        self._upgrade_button_label = pyglet.text.Label(
            "Upgrade",
            x=upgrade_button_x + upgrade_button_width / 2,
            y=upgrade_button_y + upgrade_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=12,
            color=(230, 230, 230, 255),
            batch=self.ui_batch,
        )
        self._upgrade_price_label = pyglet.text.Label(
            "",
            x=upgrade_button_x + upgrade_button_width + 8,
            y=upgrade_button_y + upgrade_button_height / 2,
            anchor_x="left",
            anchor_y="center",
            font_size=12,
            color=(200, 200, 200, 255),
            batch=self.ui_batch,
        )
        self._upgrade_button_bounds = (
            upgrade_button_x,
            upgrade_button_y,
            upgrade_button_width,
            upgrade_button_height,
        )
        self._upgrade_visible = False
        self._refresh_upgrade_button()

        speed_button_height = 28
        speed_button_width = int(self._sidebar_width * 0.3)
        speed_button_y = self._sidebar_padding
        speed_title_y = speed_button_y + speed_button_height + 8
        speed_left_x = self._sidebar_x
        speed_right_x = self._sidebar_x + self._sidebar_width - speed_button_width

        self._speed_title_label = pyglet.text.Label(
            "Simulation speed",
            x=self._sidebar_x + self._sidebar_width / 2,
            y=speed_title_y,
            anchor_x="center",
            anchor_y="bottom",
            font_size=12,
            color=(200, 200, 200, 255),
            batch=self.ui_batch,
        )

        button_color = (70, 70, 74)
        button_border = (120, 120, 125)
        self._speed_down_button = pyglet.shapes.BorderedRectangle(
            speed_left_x,
            speed_button_y,
            speed_button_width,
            speed_button_height,
            color=button_color,
            border=2,
            border_color=button_border,
            batch=self.ui_batch,
        )
        self._speed_up_button = pyglet.shapes.BorderedRectangle(
            speed_right_x,
            speed_button_y,
            speed_button_width,
            speed_button_height,
            color=button_color,
            border=2,
            border_color=button_border,
            batch=self.ui_batch,
        )

        self._speed_down_label = pyglet.text.Label(
            "<<",
            x=speed_left_x + speed_button_width / 2,
            y=speed_button_y + speed_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=14,
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._speed_up_label = pyglet.text.Label(
            ">>",
            x=speed_right_x + speed_button_width / 2,
            y=speed_button_y + speed_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=14,
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )

        speed_label = SPEED_OPTIONS[self._speed_index][0]
        self._speed_value_label = pyglet.text.Label(
            speed_label,
            x=self._sidebar_x + self._sidebar_width / 2,
            y=speed_button_y + speed_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=13,
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )

        self._speed_down_bounds = (speed_left_x, speed_button_y, speed_button_width, speed_button_height)
        self._speed_up_bounds = (speed_right_x, speed_button_y, speed_button_width, speed_button_height)
        self._refresh_speed_buttons()

        self._game_over_overlay = pyglet.shapes.Rectangle(
            0,
            0,
            self.map_image.width,
            self.map_image.height,
            color=(0, 0, 0),
            batch=self.ui_batch,
        )
        self._game_over_overlay.opacity = 0
        self._game_over_label = pyglet.text.Label(
            "GAME LOST",
            x=self.map_image.width / 2,
            y=self.map_image.height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=36,
            color=(220, 30, 30, 0),
            batch=self.ui_batch,
        )
        self._game_over_label_color = (220, 30, 30)

    def _set_speed_index(self, index: int) -> None:
        clamped = max(0, min(index, len(SPEED_OPTIONS) - 1))
        if clamped == self._speed_index:
            return
        self._speed_index = clamped
        label, factor = SPEED_OPTIONS[clamped]
        self._speed_factor = factor
        self._speed_value_label.text = label
        print(f"[gui] speed set to {label} ({factor}x)")
        self._refresh_speed_buttons()

    def _refresh_speed_buttons(self) -> None:
        inactive = (50, 50, 54)
        active = (70, 70, 74)
        inactive_border = (90, 90, 94)
        active_border = (120, 120, 125)
        if self._speed_index <= 0:
            self._speed_down_button.color = inactive
            self._speed_down_button.border_color = inactive_border
        else:
            self._speed_down_button.color = active
            self._speed_down_button.border_color = active_border
        if self._speed_index >= len(SPEED_OPTIONS) - 1:
            self._speed_up_button.color = inactive
            self._speed_up_button.border_color = inactive_border
        else:
            self._speed_up_button.color = active
            self._speed_up_button.border_color = active_border

    def _refresh_tower_button(self) -> None:
        tower_def = get_tower_def(self._placement_kind)
        can_afford = self.state.bank >= tower_def.cost
        if not can_afford and self._placement_active:
            self._set_placement_active(False)
        if self._placement_active:
            self._tower_button.color = (70, 90, 70)
            self._tower_button.border_color = (120, 180, 120)
            self._tower_button.opacity = 255
            self._tower_button_sprite.color = (255, 255, 255)
            self._tower_button_enabled = True
        else:
            if can_afford:
                self._tower_button.color = (60, 60, 64)
                self._tower_button.border_color = (110, 110, 115)
                self._tower_button.opacity = 255
                self._tower_button_sprite.color = (255, 255, 255)
                self._tower_button_enabled = True
            else:
                self._tower_button.color = (40, 40, 44)
                self._tower_button.border_color = (80, 80, 85)
                self._tower_button.opacity = 170
                self._tower_button_sprite.color = (130, 130, 130)
                self._tower_button_enabled = False

    def _set_placement_active(self, active: bool) -> None:
        if active == self._placement_active:
            return
        self._placement_active = active
        self._refresh_tower_button()
        if active:
            self._set_selected_tower(None)
        if not active:
            self._tower_preview_sprite.opacity = 0
            self._placement_cell = None
            self._placement_valid = False
            return
        if self._last_mouse_pos is not None:
            self._update_placement_preview(*self._last_mouse_pos)

    def _set_game_over_overlay(self, active: bool) -> None:
        if active:
            self._game_over_overlay.opacity = 150
            self._game_over_label.color = (*self._game_over_label_color, 255)
        else:
            self._game_over_overlay.opacity = 0
            self._game_over_label.color = (*self._game_over_label_color, 0)

    def _set_window_to_screen(self) -> None:
        screen = self.window.screen
        if screen is None:
            display = pyglet.canvas.get_display()
            screen = display.get_default_screen()
        if screen is None:
            return
        self.window.set_size(screen.width, screen.height)
        self.window.set_location(screen.x, screen.y)
        self._window_width = self.window.width
        self._window_height = self.window.height
        self._update_view_transform()

    def _build_grid_overlay(self) -> None:
        grid = self.map_data.grid
        if grid <= 0:
            return
        for shape in self._grid_shapes:
            shape.delete()
        self._grid_shapes.clear()

        cells = buildable_cells(self.map_data)
        fill_color = (0, 80, 0)
        border_color = (0, 200, 0)
        for x, y in cells:
            draw_x = x + self._world_offset[0]
            draw_y = self.map_image.height - (y + grid + self._world_offset[1])
            rect = pyglet.shapes.BorderedRectangle(
                draw_x,
                draw_y,
                grid,
                grid,
                color=fill_color,
                border=1,
                border_color=border_color,
                batch=self.batch,
            )
            rect.opacity = 120
            rect.border_opacity = 200
            self._grid_shapes.append(rect)

    def _tower_draw_position(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        grid = self.map_data.grid
        draw_x = cell_x + grid * 0.5 + self._world_offset[0]
        draw_y = self.map_image.height - (cell_y + grid * 0.5 + self._world_offset[1])
        return draw_x, draw_y

    def _grid_cell_at(self, logical_x: float, logical_y: float) -> tuple[int, int] | None:
        map_x = logical_x - self._world_offset[0]
        map_y = self.map_image.height - logical_y - self._world_offset[1]
        grid = self.map_data.grid
        if map_x < 0 or map_y < 0:
            return None
        if map_x >= self.map_data.width or map_y >= self.map_data.height:
            return None
        cell_x = int(map_x // grid) * grid
        cell_y = int(map_y // grid) * grid
        if cell_x < 0 or cell_y < 0:
            return None
        if cell_x + grid > self.map_data.width or cell_y + grid > self.map_data.height:
            return None
        return cell_x, cell_y

    def _update_placement_preview(self, logical_x: float, logical_y: float) -> None:
        if not self._placement_active:
            self._tower_preview_sprite.opacity = 0
            return
        cell = self._grid_cell_at(logical_x, logical_y)
        if cell is None:
            self._tower_preview_sprite.opacity = 0
            self._placement_cell = None
            self._placement_valid = False
            return
        cell_x, cell_y = cell
        valid = can_place_tower(self.state, self.map_data, cell_x, cell_y, self._placement_kind)
        draw_x, draw_y = self._tower_draw_position(cell_x, cell_y)
        self._tower_preview_sprite.x = draw_x
        self._tower_preview_sprite.y = draw_y
        self._tower_preview_sprite.opacity = 150
        self._tower_preview_sprite.color = (255, 255, 255) if valid else (230, 70, 70)
        self._placement_cell = cell
        self._placement_valid = valid

    def _tower_at(self, logical_x: float, logical_y: float):
        cell = self._grid_cell_at(logical_x, logical_y)
        if cell is None:
            return None
        cell_x, cell_y = cell
        for tower in self.state.towers:
            if tower.cell_x == cell_x and tower.cell_y == cell_y:
                return tower
        return None

    def _handle_tower_selection(self, logical_x: float, logical_y: float) -> None:
        tower = self._tower_at(logical_x, logical_y)
        self._set_selected_tower(tower)

    def _set_selected_tower(self, tower) -> None:
        self._selected_tower = tower
        if tower is None:
            self._tower_range_shape.opacity = 0
            if self._hovered_tower_kind is None:
                self._set_info_text("")
            self._refresh_upgrade_button()
            return
        draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
        self._tower_range_shape.x = draw_x
        self._tower_range_shape.y = draw_y
        self._tower_range_shape.radius = float(tower.range)
        self._tower_range_shape.opacity = 120
        if self._hovered_tower_kind is None:
            self._set_info_text(self._tower_info_text_from_tower(tower))
        self._refresh_upgrade_button()

    def _tower_info_text_from_tower(self, tower) -> str:
        lines = [
            tower.title,
            f"DMG: {tower.damage}",
            f"RANGE: {tower.range}m",
            f"LVL: {tower.level}",
        ]
        if tower.description:
            lines.append("")
            lines.append(tower.description)
        return "\n".join(lines)

    def _tower_info_text_from_kind(self, kind: str) -> str:
        tower_def = get_tower_def(kind)
        cost_line = f"COST ${tower_def.cost}"
        if self.state.bank < tower_def.cost:
            cost_line += " (insufficient funds)"
        return (
            f"{tower_def.title}\n"
            f"{cost_line}\n"
            f"DMG: {tower_def.damage}\n"
            f"{tower_def.description}"
        )

    def _update_tower_button_hover(self, logical_x: float, logical_y: float) -> None:
        hovered_kind = None
        if _point_in_rect(logical_x, logical_y, self._tower_button_bounds):
            hovered_kind = self._placement_kind
        if hovered_kind == self._hovered_tower_kind:
            return
        self._hovered_tower_kind = hovered_kind
        if hovered_kind is not None:
            self._set_info_text(self._tower_info_text_from_kind(hovered_kind))
        else:
            if self._selected_tower is not None:
                self._set_info_text(self._tower_info_text_from_tower(self._selected_tower))
            else:
                self._set_info_text("")

    def _set_info_text(self, text: str) -> None:
        base_size = 10
        min_size = 7
        _, _, info_w, info_h = self._tower_info_bounds
        max_height = max(0, info_h - 16)
        max_width = max(0, info_w - 16)
        size = base_size
        self._tower_info_label.width = int(max_width)
        while size >= min_size:
            self._tower_info_label.font_size = size
            self._tower_info_label.text = text
            if self._tower_info_label.content_height <= max_height:
                return
            size -= 1
        self._tower_info_label.font_size = min_size
        self._tower_info_label.text = text

    def _refresh_upgrade_button(self) -> None:
        if self._selected_tower is None or self._selected_tower.level >= 10:
            self._upgrade_visible = False
        else:
            self._upgrade_visible = True
        alpha = 255 if self._upgrade_visible else 0
        self._upgrade_button.opacity = alpha
        self._upgrade_button_label.color = (
            self._upgrade_button_label.color[0],
            self._upgrade_button_label.color[1],
            self._upgrade_button_label.color[2],
            alpha,
        )
        self._upgrade_price_label.color = (
            self._upgrade_price_label.color[0],
            self._upgrade_price_label.color[1],
            self._upgrade_price_label.color[2],
            alpha,
        )
        if not self._upgrade_visible:
            self._upgrade_price_label.text = ""
            return
        upgrade_cost = int(self._selected_tower.base_cost / 2)
        can_afford = self.state.bank >= upgrade_cost
        self._upgrade_button.color = (60, 60, 64) if can_afford else (40, 40, 44)
        self._upgrade_button.border_color = (110, 110, 115) if can_afford else (80, 80, 85)
        self._upgrade_button_label.color = (230, 230, 230, 255 if can_afford else 160)
        price_color = (40, 200, 80, 255) if can_afford else (220, 60, 60, 255)
        self._upgrade_price_label.color = price_color
        self._upgrade_price_label.text = f"${upgrade_cost}"

    def _handle_upgrade_click(self) -> None:
        if self._selected_tower is None:
            return
        upgraded = upgrade_tower(self.state, self._selected_tower)
        if not upgraded:
            return
        self._set_selected_tower(self._selected_tower)
        self._refresh_upgrade_button()

    def _update_view_transform(self) -> None:
        if self._logical_width <= 0 or self._logical_height <= 0:
            self._view_scale = 1.0
            self._view_offset_x = 0.0
            self._view_offset_y = 0.0
            return
        scale_x = self._window_width / self._logical_width
        scale_y = self._window_height / self._logical_height
        self._view_scale = min(scale_x, scale_y)
        self._view_offset_x = (self._window_width - self._logical_width * self._view_scale) / 2.0
        self._view_offset_y = (self._window_height - self._logical_height * self._view_scale) / 2.0

    def _to_logical_coords(self, x: float, y: float) -> tuple[float, float]:
        fb_w, fb_h = self.window.get_framebuffer_size()
        if fb_w > 0 and fb_h > 0:
            x *= self._window_width / fb_w
            y *= self._window_height / fb_h
        return (x - self._view_offset_x) / self._view_scale, (y - self._view_offset_y) / self._view_scale


def _point_in_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= (rx + rw) and ry <= y <= (ry + rh)


def run(map_path: str | Path = Path("data/maps/switchback.json")) -> None:
    run_dir = _next_run_dir(Path("runs"))
    log_path = run_dir / "log.txt"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    _app = SimpleGui(Path(map_path), run_dir)
    pyglet.app.run()
