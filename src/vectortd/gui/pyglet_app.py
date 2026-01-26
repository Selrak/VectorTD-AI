from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
import queue
import threading
import math
from pathlib import Path
import json
import shutil
import sys
from typing import Iterable, List

import pyglet
from pyglet import gl
from pyglet.window import key
from pyglet.math import Mat4

from .assets import load_creep_images, load_map_image, load_tower_images
from .ui_layout import BottomLayout, SidebarLayout
from ..core.model.map import MapData, load_map_json
from ..core.model.state import GameState
from ..core.model.towers import get_tower_def, list_tower_defs
from ..core.rules.creep_motion import step_creeps
from ..core.rules.tower_attack import step_towers
from ..core.rules.wave_spawner import LEVELS, maybe_auto_next_wave, start_next_wave, wave_display_info
from ..core.rules.buffs import buy_emergency_lives, buy_interest
from ..core.rules.placement import (
    buildable_cells,
    can_place_tower,
    place_tower,
    sell_tower,
    set_target_mode,
    upgrade_tower,
)
from ..core.rng import seed_state
from ..io.replay import Replay, build_state_check, load_replay
from ..ai.actions import (
    Noop,
    Place,
    Sell,
    SetMode,
    StartWave,
    Upgrade,
    action_space_spec,
    get_tower_slots,
)
from ..testing.definitions import TestDefinition, TestRunner, apply_test_definition


UI_SCALE = 2.0 / 3.0
WORLD_RENDER_SCALE = 1.5
WINDOW_MODE = "borderless"
TEXT_SCALE = 2.0 / 3.0
MAX_SHOT_LINES = 4
ROCKET_RADIUS = 2.2
SWARM_RADIUS = 1.6
ROCKET_COLOR = (255, 80, 60)
SWARM_COLOR = (255, 160, 60)
REPLAY_NOTICE_HOLD = 2.0
REPLAY_NOTICE_FADE = 1.0

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

TEST_MODE_SPEED = 0.5
BUFF_UI_KINDS = ("buffD", "buffR", "bonus_interest", "bonus_lives")

TARGET_MODE_LABELS = {
    "closest": "CLOSE",
    "hardest": "HARD",
    "weakest": "WEAK",
    "fastest": "FAST",
    "random": "RAND",
}


@dataclass
class _SamplePoint:
    x: float
    y: float
    dx: float
    dy: float


@dataclass
class _TowerButton:
    kind: str
    bounds: tuple[float, float, float, float]
    rect: pyglet.shapes.BorderedRectangle
    sprite: pyglet.sprite.Sprite | None
    label: pyglet.text.Label | None = None


@dataclass
class _TargetModeButton:
    mode: str
    bounds: tuple[float, float, float, float]
    rect: pyglet.shapes.BorderedRectangle
    label: pyglet.text.Label


def _markers_ready(map_data: MapData) -> bool:
    if not map_data.markers:
        return False
    for x, y in map_data.markers.values():
        if x is None or y is None:
            return False
    return True


def _scaled_font(size: float) -> int:
    return max(1, int(round(size * TEXT_SCALE)))


def _target_mode_label(mode: str) -> str:
    return TARGET_MODE_LABELS.get(mode, mode.upper())


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


def _run_dir_indices(base_dir: Path) -> dict[int, Path]:
    run_dirs: dict[int, Path] = {}
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            idx = int(child.name)
        except ValueError:
            continue
        run_dirs[idx] = child
    return run_dirs


def _screenshot_count(run_dir: Path) -> int:
    screenshots_dir = run_dir / "screenshots"
    if not screenshots_dir.exists():
        return 0
    return sum(1 for child in screenshots_dir.iterdir() if child.is_file())


def _cleanup_runs(base_dir: Path, *, limit: int = 50) -> None:
    run_dirs = _run_dir_indices(base_dir)
    if len(run_dirs) < limit:
        return
    indices = sorted(run_dirs)
    if not indices:
        return
    keep: set[int] = set()
    if 0 in run_dirs:
        keep.add(0)
    last_idx = indices[-1]
    keep.add(last_idx)
    pair_candidates = [idx for idx in indices if idx not in keep]
    if len(pair_candidates) % 2 == 1:
        keep.add(pair_candidates[-1])
        pair_candidates = pair_candidates[:-1]
    for first_idx, second_idx in zip(pair_candidates[0::2], pair_candidates[1::2]):
        first_count = _screenshot_count(run_dirs[first_idx])
        second_count = _screenshot_count(run_dirs[second_idx])
        if first_count > second_count:
            keep.add(first_idx)
        else:
            keep.add(second_idx)
    for idx, path in run_dirs.items():
        if idx not in keep:
            shutil.rmtree(path)


def _next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_runs(base_dir, limit=50)
    run_dirs = _run_dir_indices(base_dir)
    max_idx = max(run_dirs, default=-1)
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
    def __init__(
        self,
        map_path: Path,
        run_dir: Path,
        *,
        test_definition: TestDefinition | None = None,
        test_speed: float = TEST_MODE_SPEED,
        replay: Replay | None = None,
        replay_path: Path | None = None,
    ) -> None:
        self.state = GameState()
        self.map_data = load_map_json(map_path)
        self._map_path = Path(map_path).resolve()

        self.map_image = load_map_image(self.map_data.name)
        self._ui_scale = UI_SCALE
        self._world_render_scale = WORLD_RENDER_SCALE
        self._world_offset = (
            max(0.0, (self.map_image.width - self.map_data.width) / 2.0),
            max(0.0, (self.map_image.height - self.map_data.height) / 2.0),
        )
        self._sidebar_width = 220
        self._sidebar_padding = 12
        self._sidebar_x = self.map_image.width
        self._logical_width = self.map_image.width + self._sidebar_width
        self._logical_height = self.map_image.height
        self._ui_layout_height = self._logical_height / max(0.001, self._ui_scale)
        self._window_width = self._logical_width
        self._window_height = self._logical_height
        self._base_view_scale = 1.0
        self._base_view_offset_x = 0.0
        self._base_view_offset_y = 0.0
        self._ui_view_scale = 1.0
        self._ui_view_offset_x = 0.0
        self._ui_view_offset_y = 0.0
        self._world_view_scale = 1.0
        self._world_view_offset_x = 0.0
        self._world_view_offset_y = 0.0

        if not _markers_ready(self.map_data):
            self.map_data.markers = _fallback_markers(self.map_data)

        self._test_mode = test_definition is not None
        self._replay = replay
        self._replay_path = replay_path.resolve() if replay_path is not None else None
        self._replay_paths: list[Path] = []
        self._replay_index: int | None = None
        self._replay_spec = action_space_spec(self.map_data) if replay is not None else None
        self._replay_wave_index = 0
        self._replay_in_wave = False
        self._replay_done = False
        self._replay_waiting_send = False
        self._auto_send_armed = False
        self._next_wave_button = None
        self._next_wave_label = None
        self._auto_wave_button = None
        self._auto_wave_label = None
        self._replay_checks = None
        self._replay_wave_ticks = 0
        self._replay_active_wave_index = None
        self._replay_check_notice = False
        self._fixed_accum = 0.0
        self._replay_playing = False
        self._replay_pause_at_end = False
        self._replay_notice_label = None
        self._replay_notice_time: float | None = None
        self._test_runner: TestRunner | None = None
        self._test_result: str | None = None
        if test_definition is not None:
            apply_test_definition(self.state, self.map_data, test_definition)
            self._test_runner = TestRunner(test_definition, self.state)
            tower_desc = ", ".join(
                f"{tower.kind}@{tower.cell_x},{tower.cell_y}+{tower.upgrades}"
                for tower in test_definition.towers
            )
            if not tower_desc:
                tower_desc = "none"
            print(
                "[test] mode=on "
                f"name={test_definition.name} map={self.map_data.name} "
                f"bank={self.state.bank} lives={self.state.lives} wave={self.state.level} "
                f"towers=[{tower_desc}] speed={test_speed}x"
            )
        elif replay is not None:
            self._refresh_replay_paths()
            if self._replay_path is not None:
                self._log_replay_open(self._replay_path)
            self._replay_checks = replay.state_hashes
            self.state.auto_level = False
            seed_state(self.state, replay.seed)
            print(
                "[replay] mode=on "
                f"map={self.map_data.name} waves={len(replay.waves)} seed={replay.seed}"
            )
            self._prepare_next_replay_wave()
        else:
            self._load_auto_send_state()
            self._refresh_auto_wave_button()

        fullscreen = WINDOW_MODE == "exclusive"
        window_style = pyglet.window.Window.WINDOW_STYLE_BORDERLESS if WINDOW_MODE == "borderless" else pyglet.window.Window.WINDOW_STYLE_DEFAULT
        self.window = pyglet.window.Window(
            width=self._window_width,
            height=self._window_height,
            caption=f"VectorTD - {self.map_data.name}",
            fullscreen=fullscreen,
            style=window_style,
        )
        if self.window.fullscreen:
            self._window_width = self.window.width
            self._window_height = self.window.height
            self._update_view_transform()
        else:
            self._set_window_to_screen()
        self.batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()
        self.map_sprite = pyglet.sprite.Sprite(self.map_image, x=0, y=0, batch=self.batch)
        self.map_sprite.scale = self._world_render_scale
        self._grid_shapes: list[pyglet.shapes.Rectangle] = []
        self._build_grid_overlay()

        self.tower_images = load_tower_images()
        self._tower_defs = list_tower_defs()
        if not self._tower_defs:
            raise ValueError("No tower definitions available")
        self.tower_sprites: List[pyglet.sprite.Sprite] = []
        self._tower_sprite_ids: list[int] = []
        self._tower_shot_lines: list[list[pyglet.shapes.Line]] = []
        self._tower_shot_jitter: list[tuple[int, list[tuple[float, float, float, float]]] | None] = []
        self._tower_level_tags: list[tuple[pyglet.shapes.BorderedRectangle, pyglet.text.Label]] = []
        self._tower_level_tag_base = 12
        self._tower_level_font_base = _scaled_font(9)
        self._placement_active = False
        self._placement_kind = self._tower_defs[0].kind
        self._placement_cell: tuple[int, int] | None = None
        self._placement_valid = False
        self._tower_preview_sprite = pyglet.sprite.Sprite(
            self.tower_images[self._placement_kind],
            x=-1000,
            y=-1000,
            batch=self.batch,
        )
        self._tower_preview_sprite.scale = self._world_render_scale
        self._tower_preview_sprite.opacity = 0
        self._last_mouse_pos: tuple[float, float] | None = None
        self._selected_tower = None
        self._hovered_tower_kind: str | None = None
        self._tower_buttons: list[_TowerButton] = []
        self._tower_button_enabled: dict[str, bool] = {}
        self._target_mode_buttons: list[_TargetModeButton] = []
        self._target_mode_button_modes: tuple[str, ...] = ()
        self._target_mode_row_bounds = (0.0, 0.0, 0.0, 0.0)
        self._tower_range_shape = pyglet.shapes.Circle(
            -1000,
            -1000,
            radius=self._world_len(1),
            color=(80, 200, 120),
            batch=self.batch,
        )
        self._tower_range_shape.opacity = 0
        self._placement_range_shape = pyglet.shapes.Circle(
            -1000,
            -1000,
            radius=self._world_len(1),
            color=(80, 200, 120),
            batch=self.batch,
        )
        self._placement_range_shape.opacity = 0
        self._next_wave_button = None
        self._next_wave_label = None
        self._auto_wave_button = None
        self._auto_wave_label = None
        self._next_wave_bounds = (0.0, 0.0, 0.0, 0.0)
        self._auto_wave_bounds = (0.0, 0.0, 0.0, 0.0)
        self._wave_now_sprite = None
        self._wave_now_label = None
        self._wave_next_sprite = None
        self._wave_next_label = None
        self._wave_icon_size = 0.0

        self.creep_images = load_creep_images()
        self.creep_sprites: List[pyglet.sprite.Sprite] = []
        self._creep_sprite_by_id: dict[int, pyglet.sprite.Sprite] = {}
        self._rocket_shapes: dict[int, pyglet.shapes.Circle] = {}
        self._sync_creep_sprites()

        self._capture_every = 50
        self._tick_count = 0
        self._tick_accum = 0.0
        self._capture_requests: deque[int] = deque()
        self._capture_tasks: queue.Queue[tuple[int, "pyglet.image.ImageData"]] = queue.Queue()
        self._screenshot_dir = run_dir / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()

        target_speed = float(test_speed if self._test_mode else 1.0)
        self._speed_index = next(
            (i for i, (_, factor) in enumerate(SPEED_OPTIONS) if factor == target_speed),
            0,
        )
        if self._test_mode and SPEED_OPTIONS[self._speed_index][1] != target_speed:
            print(f"[test] speed {target_speed}x not found, using {SPEED_OPTIONS[self._speed_index][1]}x")
        self._speed_factor = SPEED_OPTIONS[self._speed_index][1]

        self._last_lives = self.state.lives
        self._last_bank = self.state.bank
        self._last_ups = self.state.ups
        self._last_interest = self.state.interest
        self._build_sidebar_ui()

        self.window.push_handlers(
            on_draw=self.on_draw,
            on_resize=self.on_resize,
            on_mouse_press=self.on_mouse_press,
            on_mouse_motion=self.on_mouse_motion,
            on_mouse_drag=self.on_mouse_drag,
            on_key_press=self.on_key_press,
            on_close=self.on_close,
        )
        self._apply_viewport()
        self._apply_projection()
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _auto_send_state_path(self) -> Path:
        root = Path(__file__).resolve().parents[3]
        return root / "runs" / "gui_state.json"

    def _load_auto_send_state(self) -> None:
        path = self._auto_send_state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.state.auto_level = bool(data.get("auto_send", False))

    def _save_auto_send_state(self) -> None:
        path = self._auto_send_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"auto_send": bool(self.state.auto_level)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _sync_creep_sprites(self) -> None:
        creeps = self.state.creeps
        creep_ids = {id(c) for c in creeps}
        removed = 0
        for creep_id, sprite in list(self._creep_sprite_by_id.items()):
            if creep_id not in creep_ids:
                sprite.delete()
                del self._creep_sprite_by_id[creep_id]
                removed += 1
        if removed:
            print(f"[gui] removed {removed} creep sprites (creeps={len(creeps)})")

        new_sprites: list[pyglet.sprite.Sprite] = []
        added = 0
        for creep in creeps:
            creep_id = id(creep)
            sprite = self._creep_sprite_by_id.get(creep_id)
            img = self.creep_images.get(creep.type_id, self.creep_images[1])
            if sprite is None:
                draw_x = creep.x + self._world_offset[0]
                draw_y = self.map_image.height - (creep.y + self._world_offset[1])
                sprite = pyglet.sprite.Sprite(
                    img,
                    x=draw_x,
                    y=draw_y,
                    batch=self.batch,
                )
                sprite.scale = self._world_render_scale
                self._creep_sprite_by_id[creep_id] = sprite
                added += 1
            else:
                if sprite.image is not img:
                    sprite.image = img
            sprite.x = self._world_draw_x(creep.x)
            sprite.y = self._world_draw_y(creep.y)
            new_sprites.append(sprite)
        self.creep_sprites = new_sprites
        if added:
            print(f"[gui] added {added} creep sprites (creeps={len(creeps)})")

    def _sync_rocket_shapes(self) -> None:
        rockets = getattr(self.state, "rockets", [])
        rocket_ids = {id(rocket) for rocket in rockets}
        for rocket_id, shape in list(self._rocket_shapes.items()):
            if rocket_id not in rocket_ids:
                shape.delete()
                del self._rocket_shapes[rocket_id]

        for rocket in rockets:
            rocket_id = id(rocket)
            shape = self._rocket_shapes.get(rocket_id)
            is_swarm = getattr(rocket, "kind", "rocket") != "rocket"
            color = SWARM_COLOR if is_swarm else ROCKET_COLOR
            radius = SWARM_RADIUS if is_swarm else ROCKET_RADIUS
            if shape is None:
                shape = pyglet.shapes.Circle(
                    0,
                    0,
                    radius=self._world_len(radius),
                    color=color,
                    batch=self.batch,
                )
                shape.opacity = 220
                self._rocket_shapes[rocket_id] = shape
            else:
                if shape.color != color:
                    shape.color = color
                shape.radius = self._world_len(radius)
            shape.x = self._world_draw_x(rocket.x)
            shape.y = self._world_draw_y(rocket.y)

    def _clear_tower_renderables(self) -> None:
        for sprite in self.tower_sprites:
            sprite.delete()
        for tag_rect, tag_label in self._tower_level_tags:
            tag_rect.delete()
            tag_label.delete()
        for lines in self._tower_shot_lines:
            for line in lines:
                line.delete()
        self.tower_sprites = []
        self._tower_level_tags = []
        self._tower_shot_lines = []
        self._tower_shot_jitter = []
        self._tower_sprite_ids = []

    def _sync_tower_sprites(self) -> None:
        towers = self.state.towers
        tower_ids = [id(tower) for tower in towers]
        if tower_ids != self._tower_sprite_ids:
            self._clear_tower_renderables()
        default_img = self.tower_images.get(self._placement_kind)
        if len(self.tower_sprites) < len(towers):
            for tower in towers[len(self.tower_sprites):]:
                img = self.tower_images.get(tower.kind, default_img)
                if img is None:
                    continue
                draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
                sprite = pyglet.sprite.Sprite(img, x=draw_x, y=draw_y, batch=self.batch)
                sprite.scale = self._world_render_scale
                self.tower_sprites.append(sprite)
                tag_size = self._world_len(self._tower_level_tag_base)
                tag_rect = pyglet.shapes.BorderedRectangle(
                    0,
                    0,
                    tag_size,
                    tag_size,
                    color=(0, 0, 0),
                    border=max(1, int(round(self._world_len(1)))),
                    border_color=(0, 220, 0),
                    batch=self.batch,
                )
                tag_label = pyglet.text.Label(
                    str(tower.level),
                    x=0,
                    y=0,
                    anchor_x="center",
                    anchor_y="center",
                    font_size=int(round(self._world_len(self._tower_level_font_base))),
                    color=(0, 220, 0, 255),
                    batch=self.batch,
                )
                self._tower_level_tags.append((tag_rect, tag_label))
                self._tower_sprite_ids.append(id(tower))
        if len(self._tower_shot_lines) < len(towers):
            for tower in towers[len(self._tower_shot_lines):]:
                tower_def = get_tower_def(tower.kind)
                line_width = max(1, int(round(self._world_len(1.5))))
                lines: list[pyglet.shapes.Line] = []
                for _ in range(MAX_SHOT_LINES):
                    line = pyglet.shapes.Line(
                        0,
                        0,
                        0,
                        0,
                        thickness=line_width,
                        color=tower_def.tag_color,
                        batch=self.batch,
                    )
                    line.opacity = 0
                    lines.append(line)
                self._tower_shot_lines.append(lines)
                self._tower_shot_jitter.append(None)

        for (sprite, tower), (tag_rect, tag_label) in zip(
            zip(self.tower_sprites, towers),
            self._tower_level_tags,
        ):
            draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
            sprite.x = draw_x
            sprite.y = draw_y
            tower_def = get_tower_def(tower.kind)
            tag_x, tag_y = self._tower_level_tag_position(tower.cell_x, tower.cell_y)
            tag_size = self._world_len(self._tower_level_tag_base)
            tag_rect.x = tag_x
            tag_rect.y = tag_y
            tag_rect.color = (0, 0, 0)
            tag_rect.border_color = tower_def.tag_color
            tag_label.x = tag_x + tag_size / 2
            tag_label.y = tag_y + tag_size / 2
            if tower.kind in ("buffD", "buffR"):
                tag_rect.opacity = 0
                tag_rect.border_opacity = 0
                tag_label.color = (*tower_def.tag_color, 0)
                tag_label.text = ""
            else:
                tag_rect.opacity = 255
                tag_rect.border_opacity = 255
                tag_label.color = (*tower_def.tag_color, 255)
                tag_label.text = str(tower.level)

    def _sync_tower_shots(self) -> None:
        towers = self.state.towers
        if not self._tower_shot_lines:
            return
        for slot_idx, (lines, tower) in enumerate(zip(self._tower_shot_lines, towers)):
            segments = tower.shot_segments if tower.shot_timer > 0.0 else []
            segments_to_draw = segments
            if segments and tower.kind == "red_refractor":
                # Refractor splash lines jitter in the SWF; keep it visual-only here.
                cache = self._tower_shot_jitter[slot_idx]
                seg_id = id(segments)
                if cache is None or cache[0] != seg_id:
                    jittered = [segments[0]]
                    for x1, y1, x2, y2 in segments[1:]:
                        jittered.append(
                            (
                                x1,
                                y1,
                                x2 + random.randrange(10) - 5,
                                y2 + random.randrange(10) - 5,
                            )
                        )
                    cache = (seg_id, jittered)
                    self._tower_shot_jitter[slot_idx] = cache
                segments_to_draw = cache[1]
            else:
                self._tower_shot_jitter[slot_idx] = None
            for idx, line in enumerate(lines):
                if idx >= len(segments_to_draw):
                    line.opacity = 0
                    continue
                x1, y1, x2, y2 = segments_to_draw[idx]
                line.x = self._world_draw_x(x1)
                line.y = self._world_draw_y(y1)
                line.x2 = self._world_draw_x(x2)
                line.y2 = self._world_draw_y(y2)
                line.opacity = int(getattr(tower, "shot_opacity", 200))

    def _tower_from_slot(self, tower_id: int):
        if self._replay_spec is None:
            return None
        if tower_id < 0 or tower_id >= self._replay_spec.max_towers:
            return None
        tower_slots = get_tower_slots(self.state, self._replay_spec.max_towers)
        return tower_slots[tower_id]

    def _apply_replay_action(self, action) -> None:
        spec = self._replay_spec
        if spec is None:
            return
        if isinstance(action, Noop):
            return
        if isinstance(action, Place):
            if action.tower_type < 0 or action.tower_type >= len(spec.tower_kinds):
                return
            if action.cell < 0 or action.cell >= len(spec.cell_positions):
                return
            cell_x, cell_y = spec.cell_positions[action.cell]
            tower_kind = spec.tower_kinds[action.tower_type]
            place_tower(self.state, self.map_data, cell_x, cell_y, tower_kind)
            return
        if isinstance(action, Upgrade):
            tower = self._tower_from_slot(action.tower_id)
            upgrade_tower(self.state, tower)
            return
        if isinstance(action, Sell):
            tower = self._tower_from_slot(action.tower_id)
            sell_tower(self.state, tower)
            return
        if isinstance(action, SetMode):
            tower = self._tower_from_slot(action.tower_id)
            if tower is None:
                return
            if action.mode < 0 or action.mode >= len(spec.target_modes):
                return
            set_target_mode(tower, spec.target_modes[action.mode])
            return
        if isinstance(action, StartWave):
            start_next_wave(self.state, self.map_data)
            self._replay_in_wave = True
            return

    def _prepare_next_replay_wave(self) -> None:
        if self._replay is None or self._replay_done or self._replay_waiting_send or self._replay_in_wave:
            return
        if self._replay_wave_index >= len(self._replay.waves):
            print("[replay] complete")
            self._finish_replay()
            return
        actions = self._replay.waves[self._replay_wave_index]
        for action in actions:
            if isinstance(action, StartWave):
                break
            self._apply_replay_action(action)
        self._verify_replay_state("pre", self._replay_wave_index, wave_ticks=0)
        self._replay_waiting_send = True

    def _finish_replay(self) -> None:
        if self._replay is None:
            return
        self._replay_done = True
        self._replay_playing = False
        self._replay_waiting_send = False
        self._replay_in_wave = False
        self._replay_active_wave_index = None
        self._refresh_replay_controls()

    def _replay_finished(self) -> bool:
        return self._replay is not None and (self._replay_done or self.state.game_over)

    def _refresh_replay_paths(self) -> None:
        if self._replay_path is None:
            self._replay_paths = []
            self._replay_index = None
            return
        base_dir = self._replay_path.parent
        try:
            paths = [
                path.resolve()
                for path in base_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".json"
            ]
        except OSError:
            paths = []
        if not paths:
            paths = [self._replay_path]
        if self._replay_path not in paths:
            paths.append(self._replay_path)
        self._replay_paths = sorted(set(paths))
        self._replay_index = self._replay_paths.index(self._replay_path)

    def _log_replay_open(self, replay_path: Path) -> None:
        try:
            resolved = replay_path.resolve()
        except OSError:
            resolved = replay_path
        print(f"[replay] opened {resolved}")

    def _show_replay_notice(self, label: str) -> None:
        if self._replay_notice_label is None:
            return
        self._replay_notice_label.text = label
        self._replay_notice_time = 0.0
        self._update_replay_notice_position()
        self._replay_notice_label.color = (
            self._replay_notice_label.color[0],
            self._replay_notice_label.color[1],
            self._replay_notice_label.color[2],
            255,
        )

    def _update_replay_notice(self, dt: float) -> None:
        if self._replay_notice_label is None or self._replay_notice_time is None:
            return
        self._replay_notice_time += max(0.0, dt)
        if self._replay_notice_time <= REPLAY_NOTICE_HOLD:
            alpha = 255
        elif self._replay_notice_time <= (REPLAY_NOTICE_HOLD + REPLAY_NOTICE_FADE):
            fade_t = (self._replay_notice_time - REPLAY_NOTICE_HOLD) / max(0.001, REPLAY_NOTICE_FADE)
            alpha = int(round(255 * (1.0 - fade_t)))
        else:
            alpha = 0
            self._replay_notice_time = None
        if alpha <= 0:
            self._replay_notice_label.color = (
                self._replay_notice_label.color[0],
                self._replay_notice_label.color[1],
                self._replay_notice_label.color[2],
                0,
            )
            if self._replay_notice_time is None:
                self._replay_notice_label.text = ""
            return
        self._replay_notice_label.color = (
            self._replay_notice_label.color[0],
            self._replay_notice_label.color[1],
            self._replay_notice_label.color[2],
            alpha,
        )

    def _update_replay_notice_position(self) -> None:
        if self._replay_notice_label is None:
            return
        center_x, center_y = self._to_ui_coords(self._window_width / 2, self._window_height / 2)
        self._replay_notice_label.x = center_x
        self._replay_notice_label.y = center_y

    def _reset_map_assets(self, map_path: Path) -> None:
        self._map_path = map_path.resolve()
        self.map_data = load_map_json(map_path)
        if not _markers_ready(self.map_data):
            self.map_data.markers = _fallback_markers(self.map_data)
        self.map_image = load_map_image(self.map_data.name)
        self._world_offset = (
            max(0.0, (self.map_image.width - self.map_data.width) / 2.0),
            max(0.0, (self.map_image.height - self.map_data.height) / 2.0),
        )
        self._sidebar_x = self.map_image.width
        self._logical_width = self.map_image.width + self._sidebar_width
        self._logical_height = self.map_image.height
        if self.window is not None:
            self.window.set_caption(f"VectorTD - {self.map_data.name}")
        self._update_view_transform()
        self._apply_viewport()
        self._apply_projection()

        self.batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()
        self.map_sprite = pyglet.sprite.Sprite(self.map_image, x=0, y=0, batch=self.batch)
        self.map_sprite.scale = self._world_render_scale
        self._grid_shapes = []
        self._build_grid_overlay()

        self.tower_sprites = []
        self._tower_sprite_ids = []
        self._tower_shot_lines = []
        self._tower_level_tags = []
        self.creep_sprites = []
        self._creep_sprite_by_id = {}
        self._rocket_shapes = {}

        preview_image = self.tower_images.get(self._placement_kind)
        if preview_image is None:
            preview_image = next(iter(self.tower_images.values()))
        self._tower_preview_sprite = pyglet.sprite.Sprite(
            preview_image,
            x=-1000,
            y=-1000,
            batch=self.batch,
        )
        self._tower_preview_sprite.scale = self._world_render_scale
        self._tower_preview_sprite.opacity = 0
        self._tower_range_shape = pyglet.shapes.Circle(
            -1000,
            -1000,
            radius=self._world_len(1),
            color=(80, 200, 120),
            batch=self.batch,
        )
        self._tower_range_shape.opacity = 0
        self._placement_range_shape = pyglet.shapes.Circle(
            -1000,
            -1000,
            radius=self._world_len(1),
            color=(80, 200, 120),
            batch=self.batch,
        )
        self._placement_range_shape.opacity = 0

        self._tower_buttons = []
        self._tower_button_enabled = {}
        self._target_mode_buttons = []
        self._target_mode_button_modes = ()
        self._target_mode_row_bounds = (0.0, 0.0, 0.0, 0.0)
        self._build_sidebar_ui()
        self._update_replay_notice_position()

    def _reset_replay_state(
        self,
        replay: Replay,
        *,
        replay_path: Path | None,
        start_playing: bool,
        show_notice: bool,
    ) -> None:
        self._test_mode = False
        self._test_runner = None
        self._test_result = None
        self._auto_send_armed = False
        self.state = GameState()
        self.state.auto_level = False
        if self._tower_defs:
            self._placement_kind = self._tower_defs[0].kind
        self._placement_active = False
        self._placement_cell = None
        self._placement_valid = False
        self._last_mouse_pos = None
        self._selected_tower = None
        self._hovered_tower_kind = None
        self._replay = replay
        if replay_path is not None:
            self._replay_path = replay_path.resolve()
            self._refresh_replay_paths()

        map_path = _resolve_replay_map_path(replay.map_path)
        self._reset_map_assets(map_path)

        self._replay_spec = action_space_spec(self.map_data)
        self._replay_wave_index = 0
        self._replay_in_wave = False
        self._replay_done = False
        self._replay_waiting_send = False
        self._replay_checks = replay.state_hashes
        self._replay_wave_ticks = 0
        self._replay_active_wave_index = None
        self._replay_check_notice = False
        self._fixed_accum = 0.0
        self._tick_count = 0
        self._tick_accum = 0.0
        self._capture_requests.clear()

        seed_state(self.state, replay.seed)
        self._prepare_next_replay_wave()

        self._last_lives = self.state.lives
        self._last_bank = self.state.bank
        self._last_ups = self.state.ups
        self._last_interest = self.state.interest
        if self._lives_label is not None:
            self._lives_label.text = f"Lives: {self.state.lives}"
        if self._money_label is not None:
            self._money_label.text = f"Money: ${self.state.bank}"
        if self._ups_label is not None:
            self._ups_label.text = f"Points bonus: {self.state.ups}"
        if self._interest_label is not None:
            self._interest_label.text = f"Interest: {self.state.interest}%"
        self._set_selected_tower(None)
        self._set_info_text("", None)
        self._refresh_tower_buttons()
        self._refresh_upgrade_button()
        self._refresh_sell_button()
        self._refresh_target_mode_buttons()
        self._refresh_wave_info()

        self._replay_playing = start_playing and not self._replay_done
        self._refresh_replay_controls()
        self._sync_creep_sprites()
        self._sync_rocket_shapes()
        self._sync_tower_sprites()
        self._sync_tower_shots()

        if start_playing and not self._replay_in_wave:
            self._start_prepared_replay_wave()
        if show_notice and self._replay_path is not None:
            self._show_replay_notice(self._replay_path.name)

    def _restart_replay(self, *, start_playing: bool) -> None:
        if self._replay is None:
            return
        self._reset_replay_state(
            self._replay,
            replay_path=self._replay_path,
            start_playing=start_playing,
            show_notice=False,
        )

    def _load_replay_from_path(
        self,
        replay_path: Path,
        *,
        start_playing: bool,
        show_notice: bool,
    ) -> None:
        try:
            replay = load_replay(replay_path)
        except Exception as exc:
            print(f"[replay] load failed: {replay_path} ({exc})")
            return
        self._log_replay_open(replay_path)
        self._reset_replay_state(
            replay,
            replay_path=replay_path,
            start_playing=start_playing,
            show_notice=show_notice,
        )

    def _navigate_replay(self, delta: int) -> None:
        if self._replay is None or self._replay_path is None:
            return
        self._refresh_replay_paths()
        if not self._replay_paths or self._replay_index is None:
            return
        new_index = self._replay_index + delta
        if new_index < 0 or new_index >= len(self._replay_paths):
            return
        new_path = self._replay_paths[new_index]
        if new_path == self._replay_path:
            return
        start_playing = self._replay_playing
        self._load_replay_from_path(new_path, start_playing=start_playing, show_notice=True)

    def _start_prepared_replay_wave(self) -> None:
        if self._replay is None or self._replay_done or self._replay_in_wave:
            return
        if not self._replay_waiting_send:
            self._prepare_next_replay_wave()
        if not self._replay_waiting_send:
            return
        actions = self._replay.waves[self._replay_wave_index]
        saw_start = False
        self._replay_active_wave_index = self._replay_wave_index
        self._replay_wave_ticks = 0
        for action in actions:
            if isinstance(action, StartWave):
                self._apply_replay_action(action)
                saw_start = True
                break
        if not saw_start:
            print(f"[replay] missing StartWave in wave {self._replay_wave_index}")
            self._replay_active_wave_index = None
            self._finish_replay()
            return
        self._replay_waiting_send = False
        self._replay_wave_index += 1
        self._fixed_accum = 0.0

    def _verify_replay_state(self, phase: str, wave_index: int, *, wave_ticks: int) -> None:
        if self._replay_checks is None:
            if not self._replay_check_notice:
                print("Replay verif failed : replay has no state checks")
                self._replay_check_notice = True
            return
        if not isinstance(self._replay_checks, list):
            print("Replay verif failed : replay checks format invalid")
            return
        if wave_index < 0 or wave_index >= len(self._replay_checks):
            print(f"Replay verif failed : missing checks for wave {wave_index} ({phase})")
            return
        entry = self._replay_checks[wave_index]
        if not isinstance(entry, dict):
            print(f"Replay verif failed : invalid check entry for wave {wave_index}")
            return
        expected = entry.get(phase)
        if not isinstance(expected, dict):
            print(f"Replay verif failed : missing {phase} check for wave {wave_index}")
            return
        if self._replay_spec is None:
            print("Replay verif failed : missing replay action spec")
            return
        actual = build_state_check(self.state, self.map_data, self._replay_spec, wave_ticks=wave_ticks)
        if expected.get("hash") != actual.get("hash"):
            print(
                "Replay verif failed : state hash differs "
                f"(wave {wave_index} {phase})"
            )
        expected_sections = expected.get("section_hashes")
        actual_sections = actual.get("section_hashes")
        expected_ordering = expected.get("ordering_hashes")
        actual_ordering = actual.get("ordering_hashes")
        if isinstance(expected_sections, dict) and isinstance(actual_sections, dict):
            for key, expected_value in expected_sections.items():
                if actual_sections.get(key) == expected_value:
                    continue
                if (
                    isinstance(expected_ordering, dict)
                    and isinstance(actual_ordering, dict)
                    and key in {"creeps", "towers", "pulses", "rockets"}
                ):
                    order_key = f"{key}_sorted"
                    if (
                        expected_ordering.get(order_key) is not None
                        and expected_ordering.get(order_key) == actual_ordering.get(order_key)
                    ):
                        print(
                            "Replay verif failed : "
                            f"{key} order differs (content matches sorted) "
                            f"(wave {wave_index} {phase})"
                        )
                        continue
                print(
                    "Replay verif failed : "
                    f"section hash differs ({key}) "
                    f"(wave {wave_index} {phase})"
                )
        if isinstance(expected_ordering, dict) and isinstance(actual_ordering, dict):
            for key, expected_value in expected_ordering.items():
                if actual_ordering.get(key) != expected_value:
                    print(
                        "Replay verif failed : "
                        f"ordering hash differs ({key}) "
                        f"(wave {wave_index} {phase})"
                    )
        if expected.get("rng_state") != actual.get("rng_state"):
            print(
                "Replay verif failed : rng_state differs "
                f"(wave {wave_index} {phase} expected={expected.get('rng_state')} got={actual.get('rng_state')})"
            )
        if expected.get("rng_calls") != actual.get("rng_calls"):
            print(
                "Replay verif failed : rng_calls differs "
                f"(wave {wave_index} {phase} expected={expected.get('rng_calls')} got={actual.get('rng_calls')})"
            )
        if expected.get("slot_trace") != actual.get("slot_trace"):
            print(f"Replay verif failed : slot trace differs (wave {wave_index} {phase})")
        if expected.get("wave_ticks") != actual.get("wave_ticks"):
            print(
                "Replay verif failed : wave ticks differs "
                f"(wave {wave_index} {phase} expected={expected.get('wave_ticks')} got={actual.get('wave_ticks')})"
            )

    def update(self, dt: float) -> None:
        if self._test_result is not None:
            self._set_game_over_overlay(True, test_result=self._test_result)
            return
        self._update_replay_notice(dt)
        frame_steps = max(0.0, dt) * 60.0 * self._speed_factor
        if self._replay is not None:
            if self._replay_in_wave and self._replay_playing:
                self._fixed_accum += frame_steps
        else:
            self._fixed_accum += frame_steps
        whole_steps = int(self._fixed_accum)
        self._fixed_accum -= whole_steps
        wave_step_count = 0
        for _ in range(whole_steps):
            step_creeps(self.state, self.map_data, dt_scale=1.0)
            step_towers(self.state, self.map_data, dt_scale=1.0)
            if self._replay_in_wave:
                wave_step_count += 1
                if not self.state.creeps:
                    break
            if self._replay is None and self.state.auto_level and self._auto_send_armed:
                maybe_auto_next_wave(self.state, self.map_data)
        if wave_step_count > 0:
            self._replay_wave_ticks += wave_step_count
        if self._test_runner is not None:
            self._test_runner.advance(frame_steps)
            if self._test_runner.status != "running":
                self._test_result = self._test_runner.status
                self.state.game_over = True
                self.state.game_won = self._test_result == "passed"
        tick_steps = float(whole_steps)
        if self._replay is not None and not self._replay_in_wave:
            tick_steps = 0.0
        self._tick_accum += tick_steps
        new_ticks = int(self._tick_accum)
        if new_ticks > 0:
            self._tick_accum -= new_ticks
            old_count = self._tick_count
            self._tick_count += new_ticks
            if self._capture_every > 0:
                next_cap = ((old_count // self._capture_every) + 1) * self._capture_every
                for tick in range(next_cap, self._tick_count + 1, self._capture_every):
                    self._capture_requests.append(tick)
        if self.state.lives != self._last_lives:
            self._last_lives = self.state.lives
            self._lives_label.text = f"Lives: {self.state.lives}"
        if self.state.bank != self._last_bank:
            self._last_bank = self.state.bank
            self._money_label.text = f"Money: ${self.state.bank}"
            self._refresh_tower_buttons()
            self._refresh_upgrade_button()
            self._refresh_sell_button()
        if self.state.ups != self._last_ups:
            self._last_ups = self.state.ups
            self._ups_label.text = f"Points bonus: {self.state.ups}"
            self._refresh_tower_buttons()
        if self.state.interest != self._last_interest:
            self._last_interest = self.state.interest
            self._interest_label.text = f"Interest: {self.state.interest}%"
        if self.state.lives <= 0 and not self.state.game_over:
            self.state.game_over = True
            self.state.game_won = False
        if not self.state.game_over:
            self._maybe_finish_game()
        if self._replay is not None and self.state.game_over and not self._replay_done:
            self._finish_replay()
        if self._replay is not None and not self._replay_done and not self.state.game_over:
            if self._replay_in_wave and not self.state.creeps:
                self._replay_in_wave = False
                self._fixed_accum = 0.0
                if self._replay_active_wave_index is not None:
                    self._verify_replay_state(
                        "post",
                        self._replay_active_wave_index,
                        wave_ticks=self._replay_wave_ticks,
                    )
                    self._replay_active_wave_index = None
                if self._replay_pause_at_end:
                    self._replay_playing = False
                    self._refresh_replay_controls()
            if not self._replay_in_wave:
                self._prepare_next_replay_wave()
                if self._replay_playing and not self._replay_pause_at_end:
                    self._start_prepared_replay_wave()
        self._refresh_wave_info()
        self._refresh_target_mode_buttons()
        if self._test_result is not None:
            self._set_game_over_overlay(True, test_result=self._test_result)
        else:
            self._set_game_over_overlay(self.state.game_over)
        self._sync_creep_sprites()
        self._sync_rocket_shapes()
        self._sync_tower_sprites()
        self._sync_tower_shots()

    def on_draw(self) -> None:
        self.window.clear()
        self._apply_world_view()
        self.batch.draw()
        self._apply_ui_view()
        self.ui_batch.draw()
        if self._capture_requests:
            tick = self._capture_requests.popleft()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image = buffer.get_image_data()
            self._capture_tasks.put((tick, image))

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

    def _apply_view(self, scale: float, offset_x: float, offset_y: float) -> None:
        self.window.view = Mat4().translate(
            (offset_x, offset_y, 0.0),
        ).scale(
            (scale, scale, 1.0),
        )

    def _apply_world_view(self) -> None:
        self._apply_view(self._world_view_scale, self._world_view_offset_x, self._world_view_offset_y)

    def _apply_ui_view(self) -> None:
        self._apply_view(self._ui_view_scale, self._ui_view_offset_x, self._ui_view_offset_y)

    def on_resize(self, width: int, height: int) -> None:
        self._window_width = width
        self._window_height = height
        self._update_view_transform()
        self._apply_viewport()
        self._apply_projection()
        self._update_replay_notice_position()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        world_x, world_y = self._to_world_coords(x, y)
        ui_x, ui_y = self._to_ui_coords(x, y)
        in_down = _point_in_rect(ui_x, ui_y, self._speed_down_bounds)
        in_up = _point_in_rect(ui_x, ui_y, self._speed_up_bounds)
        in_send = _point_in_rect(ui_x, ui_y, self._next_wave_bounds)
        in_auto = _point_in_rect(ui_x, ui_y, self._auto_wave_bounds)
        tower_kind = self._tower_button_kind_at(ui_x, ui_y)
        in_upgrade = self._upgrade_visible and _point_in_rect(ui_x, ui_y, self._upgrade_button_bounds)
        in_sell = self._sell_visible and _point_in_rect(ui_x, ui_y, self._sell_button_bounds)
        target_mode = self._target_mode_button_at(ui_x, ui_y)
        print(
            "[gui] click "
            f"raw=({x},{y}) world=({world_x:.1f},{world_y:.1f}) ui=({ui_x:.1f},{ui_y:.1f}) "
            f"button={button} down={in_down} up={in_up}"
        )
        if in_upgrade:
            self._handle_upgrade_click()
            return
        if in_sell:
            self._handle_sell_click()
            return
        if target_mode is not None:
            self._handle_target_mode_click(target_mode)
            return
        if in_send:
            self._handle_next_wave_click()
            return
        if in_auto:
            self._toggle_auto_wave()
            return
        if tower_kind is not None:
            if not self._tower_button_enabled.get(tower_kind, True):
                return
            if tower_kind == "bonus_interest":
                buy_interest(self.state)
                return
            if tower_kind == "bonus_lives":
                buy_emergency_lives(self.state)
                return
            if self._placement_active and self._placement_kind == tower_kind:
                self._set_placement_active(False)
            else:
                self._placement_kind = tower_kind
                self._tower_preview_sprite.image = self.tower_images[self._placement_kind]
                if self._placement_active:
                    self._refresh_tower_buttons()
                    if self._last_mouse_pos is not None:
                        self._update_placement_preview(*self._last_mouse_pos)
                else:
                    self._set_placement_active(True)
            return
        if in_down:
            self._set_speed_index(self._speed_index - 1)
        elif in_up:
            self._set_speed_index(self._speed_index + 1)
        elif self._placement_active:
            cell = self._grid_cell_at(world_x, world_y)
            if cell is not None:
                cell_x, cell_y = cell
                if can_place_tower(self.state, self.map_data, cell_x, cell_y, self._placement_kind):
                    place_tower(self.state, self.map_data, cell_x, cell_y, self._placement_kind)
                    self._sync_tower_sprites()
                    self._set_placement_active(False)
            self._update_placement_preview(world_x, world_y)
        else:
            self._handle_tower_selection(world_x, world_y)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        world_x, world_y = self._to_world_coords(x, y)
        ui_x, ui_y = self._to_ui_coords(x, y)
        self._last_mouse_pos = (world_x, world_y)
        self._update_placement_preview(world_x, world_y)
        self._update_tower_button_hover(ui_x, ui_y)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        world_x, world_y = self._to_world_coords(x, y)
        ui_x, ui_y = self._to_ui_coords(x, y)
        self._last_mouse_pos = (world_x, world_y)
        self._update_placement_preview(world_x, world_y)
        self._update_tower_button_hover(ui_x, ui_y)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        alt_mask = getattr(key, "MOD_ALT", 0) | getattr(key, "MOD_OPTION", 0)
        if symbol in (key.PAGEUP, key.LEFT):
            self._navigate_replay(-1)
            return
        if symbol in (key.PAGEDOWN, key.RIGHT):
            self._navigate_replay(1)
            return
        if symbol == key.UP:
            self._set_speed_index(self._speed_index + 1)
            return
        if symbol == key.DOWN:
            self._set_speed_index(self._speed_index - 1)
            return
        if symbol == key.SPACE and (modifiers & alt_mask):
            self._toggle_auto_wave()
            return
        if symbol == key.SPACE:
            self._handle_next_wave_click()
            return

    def on_close(self) -> None:
        if self._replay is None:
            self._save_auto_send_state()

    def _build_sidebar_ui(self) -> None:
        sidebar_height = self._ui_layout_height
        sidebar_color = (34, 36, 40)
        self._sidebar_bg = pyglet.shapes.Rectangle(
            self._sidebar_x,
            0,
            self._sidebar_width,
            sidebar_height,
            color=sidebar_color,
            batch=self.ui_batch,
        )

        bottom_layout = BottomLayout(
            x=self._sidebar_x,
            y_bottom=0,
            width=self._sidebar_width,
            padding=self._sidebar_padding,
            spacing=10,
        )
        speed_button_height = 28
        speed_button_width = int(self._sidebar_width * 0.3)
        speed_section_height = speed_button_height + 8 + 16
        speed_section_x, speed_section_y, speed_section_w, _ = bottom_layout.add_box(speed_section_height)
        speed_button_y = speed_section_y
        speed_title_y = speed_section_y + speed_button_height + 8
        speed_left_x = speed_section_x
        speed_right_x = speed_section_x + speed_section_w - speed_button_width

        self._speed_title_label = pyglet.text.Label(
            "Simulation speed",
            x=speed_section_x + speed_section_w / 2,
            y=speed_title_y,
            anchor_x="center",
            anchor_y="bottom",
            font_size=_scaled_font(12),
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
            font_size=_scaled_font(14),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._speed_up_label = pyglet.text.Label(
            ">>",
            x=speed_right_x + speed_button_width / 2,
            y=speed_button_y + speed_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=_scaled_font(14),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )

        speed_label = SPEED_OPTIONS[self._speed_index][0]
        self._speed_value_label = pyglet.text.Label(
            speed_label,
            x=speed_section_x + speed_section_w / 2,
            y=speed_button_y + speed_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=_scaled_font(13),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )

        self._speed_down_bounds = (speed_left_x, speed_button_y, speed_button_width, speed_button_height)
        self._speed_up_bounds = (speed_right_x, speed_button_y, speed_button_width, speed_button_height)
        self._refresh_speed_buttons()

        wave_button_height = 28
        wave_button_spacing = 8
        wave_row_x, wave_row_y, wave_row_w, wave_row_h = bottom_layout.add_box(wave_button_height)
        wave_button_w = max(10.0, (wave_row_w - wave_button_spacing) / 2)
        wave_button_y = wave_row_y + (wave_row_h - wave_button_height) / 2
        send_x = wave_row_x
        auto_x = wave_row_x + wave_button_w + wave_button_spacing
        self._next_wave_button = pyglet.shapes.BorderedRectangle(
            send_x,
            wave_button_y,
            wave_button_w,
            wave_button_height,
            color=(70, 70, 74),
            border=2,
            border_color=(120, 120, 125),
            batch=self.ui_batch,
        )
        next_label = "Play" if self._replay is not None else "Send next wave"
        self._next_wave_label = pyglet.text.Label(
            next_label,
            x=send_x + wave_button_w / 2,
            y=wave_button_y + wave_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._auto_wave_button = pyglet.shapes.BorderedRectangle(
            auto_x,
            wave_button_y,
            wave_button_w,
            wave_button_height,
            color=(70, 70, 74),
            border=2,
            border_color=(120, 120, 125),
            batch=self.ui_batch,
        )
        auto_label = "Pause at end" if self._replay is not None else "Auto send"
        self._auto_wave_label = pyglet.text.Label(
            auto_label,
            x=auto_x + wave_button_w / 2,
            y=wave_button_y + wave_button_height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._next_wave_bounds = (send_x, wave_button_y, wave_button_w, wave_button_height)
        self._auto_wave_bounds = (auto_x, wave_button_y, wave_button_w, wave_button_height)
        if self._replay is not None:
            self._refresh_replay_controls()
        else:
            self._refresh_auto_wave_button()

        wave_row_height = 36
        wave_icon_size = wave_row_height - 8
        self._wave_icon_size = wave_icon_size
        wave_text_pad = 8

        next_row_x, next_row_y, next_row_w, next_row_h = bottom_layout.add_box(wave_row_height)
        next_icon_x = next_row_x + wave_icon_size / 2
        next_icon_y = next_row_y + next_row_h / 2
        self._wave_next_sprite = pyglet.sprite.Sprite(
            self.creep_images[1],
            x=next_icon_x,
            y=next_icon_y,
            batch=self.ui_batch,
        )
        self._wave_next_sprite.scale = wave_icon_size / self.creep_images[1].width
        self._wave_next_label = pyglet.text.Label(
            "",
            x=next_row_x + wave_icon_size + wave_text_pad,
            y=next_row_y + next_row_h - 2,
            width=int(next_row_w - wave_icon_size - wave_text_pad),
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            font_size=_scaled_font(11),
            color=(220, 220, 220, 255),
            batch=self.ui_batch,
        )

        now_row_x, now_row_y, now_row_w, now_row_h = bottom_layout.add_box(wave_row_height)
        now_icon_x = now_row_x + wave_icon_size / 2
        now_icon_y = now_row_y + now_row_h / 2
        self._wave_now_sprite = pyglet.sprite.Sprite(
            self.creep_images[1],
            x=now_icon_x,
            y=now_icon_y,
            batch=self.ui_batch,
        )
        self._wave_now_sprite.scale = wave_icon_size / self.creep_images[1].width
        self._wave_now_label = pyglet.text.Label(
            "",
            x=now_row_x + wave_icon_size + wave_text_pad,
            y=now_row_y + now_row_h - 2,
            width=int(now_row_w - wave_icon_size - wave_text_pad),
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            font_size=_scaled_font(11),
            color=(220, 220, 220, 255),
            batch=self.ui_batch,
        )
        self._refresh_wave_info()

        layout = SidebarLayout(
            x=self._sidebar_x,
            y_top=sidebar_height,
            width=self._sidebar_width,
            padding=self._sidebar_padding,
            spacing=12,
            min_y=bottom_layout.cursor_y,
        )
        stats_box_height = 36
        stats_x, stats_y, stats_w, stats_h = layout.add_box(stats_box_height)
        stats_top = stats_y + stats_h
        line1_y = stats_top
        line2_y = stats_top - 18
        self._lives_label = pyglet.text.Label(
            f"Lives: {self.state.lives}",
            x=stats_x,
            y=line1_y,
            anchor_x="left",
            anchor_y="top",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._money_label = pyglet.text.Label(
            f"Money: ${self.state.bank}",
            x=stats_x + stats_w,
            y=line1_y,
            anchor_x="right",
            anchor_y="top",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._ups_label = pyglet.text.Label(
            f"Points bonus: {self.state.ups}",
            x=stats_x,
            y=line2_y,
            anchor_x="left",
            anchor_y="top",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        self._interest_label = pyglet.text.Label(
            f"Interest: {self.state.interest}%",
            x=stats_x + stats_w,
            y=line2_y,
            anchor_x="right",
            anchor_y="top",
            font_size=_scaled_font(12),
            color=(240, 240, 240, 255),
            batch=self.ui_batch,
        )
        tower_columns = 4
        tower_spacing = 6
        tower_button_size = int((layout.width - tower_spacing * (tower_columns - 1)) / tower_columns)
        total_buttons = len(self._tower_defs) + len(BUFF_UI_KINDS)
        tower_rows = math.ceil(total_buttons / tower_columns)
        tower_grid = layout.add_grid(
            rows=max(1, tower_rows),
            columns=tower_columns,
            cell_size=tower_button_size,
            spacing=tower_spacing,
        )
        self._tower_buttons.clear()
        button_kinds = [td.kind for td in self._tower_defs]
        button_kinds.extend(BUFF_UI_KINDS)
        for kind in button_kinds:
            tower_button_x, tower_button_y, _, _ = tower_grid.add_cell()
            rect = pyglet.shapes.BorderedRectangle(
                tower_button_x,
                tower_button_y,
                tower_button_size,
                tower_button_size,
                color=(60, 60, 64),
                border=2,
                border_color=(110, 110, 115),
                batch=self.ui_batch,
            )
            sprite = None
            label = None
            if kind in ("bonus_interest", "bonus_lives"):
                text = "+INT" if kind == "bonus_interest" else "+LIFE"
                label = pyglet.text.Label(
                    text,
                    x=tower_button_x + tower_button_size / 2,
                    y=tower_button_y + tower_button_size / 2,
                    anchor_x="center",
                    anchor_y="center",
                    font_size=_scaled_font(12),
                    color=(240, 240, 240, 255),
                    batch=self.ui_batch,
                )
            else:
                tower_icon = self.tower_images[kind]
                sprite = pyglet.sprite.Sprite(
                    tower_icon,
                    x=tower_button_x + tower_button_size / 2,
                    y=tower_button_y + tower_button_size / 2,
                    batch=self.ui_batch,
                )
                icon_scale = min(
                    (tower_button_size - 12) / tower_icon.width,
                    (tower_button_size - 12) / tower_icon.height,
                )
                sprite.scale = icon_scale
            bounds = (tower_button_x, tower_button_y, tower_button_size, tower_button_size)
            self._tower_buttons.append(_TowerButton(kind, bounds, rect, sprite, label))
        self._refresh_tower_buttons()

        upgrade_row_height = 36
        target_row_height = 28
        info_height = max(
            80,
            layout.remaining_height() - upgrade_row_height - target_row_height - layout.spacing * 2,
        )
        info_x, info_y, info_w, info_h = layout.add_box(info_height)
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
            font_size=_scaled_font(10),
            color=(220, 220, 220, 255),
            batch=self.ui_batch,
        )
        self._tower_info_bonus_label = pyglet.text.Label(
            "",
            x=info_x + 8,
            y=info_y + info_h - 8,
            width=int(info_w - 16),
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            font_size=_scaled_font(10),
            color=(40, 200, 80, 0),
            batch=self.ui_batch,
        )
        self._tower_info_measure_label = pyglet.text.Label("")

        target_row_x, target_row_y, target_row_w, target_row_h = layout.add_box(target_row_height)
        self._target_mode_row_bounds = (target_row_x, target_row_y, target_row_w, target_row_h)
        self._refresh_target_mode_buttons()

        upgrade_row_x, upgrade_row_y, upgrade_row_w, upgrade_row_h = layout.add_box(upgrade_row_height)
        button_spacing = 8
        button_width = max(0.0, (upgrade_row_w - button_spacing) / 2.0)
        upgrade_button_width = button_width
        upgrade_button_height = 26
        upgrade_button_x = upgrade_row_x
        upgrade_button_y = upgrade_row_y + (upgrade_row_h - upgrade_button_height) / 2
        sell_button_x = upgrade_row_x + upgrade_button_width + button_spacing
        sell_button_y = upgrade_button_y
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
            x=upgrade_button_x + 8,
            y=upgrade_button_y + upgrade_button_height / 2,
            anchor_x="left",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(230, 230, 230, 255),
            batch=self.ui_batch,
        )
        self._upgrade_label_pos = (
            upgrade_button_x + 8,
            upgrade_button_y + upgrade_button_height / 2,
        )
        self._upgrade_price_label = pyglet.text.Label(
            "",
            x=upgrade_button_x + 8,
            y=upgrade_button_y + upgrade_button_height / 2,
            anchor_x="left",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(200, 200, 200, 255),
            batch=self.ui_batch,
        )
        self._upgrade_button_bounds = (
            upgrade_button_x,
            upgrade_button_y,
            upgrade_button_width,
            upgrade_button_height,
        )
        self._upgrade_hidden_pos = (upgrade_button_x, upgrade_button_y)
        self._upgrade_visible = False
        self._refresh_upgrade_button()

        self._sell_button = pyglet.shapes.BorderedRectangle(
            sell_button_x,
            sell_button_y,
            upgrade_button_width,
            upgrade_button_height,
            color=(60, 60, 64),
            border=2,
            border_color=(110, 110, 115),
            batch=self.ui_batch,
        )
        self._sell_button_label = pyglet.text.Label(
            "Sell",
            x=sell_button_x + 8,
            y=sell_button_y + upgrade_button_height / 2,
            anchor_x="left",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(230, 230, 230, 255),
            batch=self.ui_batch,
        )
        self._sell_label_pos = (
            sell_button_x + 8,
            sell_button_y + upgrade_button_height / 2,
        )
        self._sell_price_label = pyglet.text.Label(
            "",
            x=sell_button_x + 8,
            y=sell_button_y + upgrade_button_height / 2,
            anchor_x="left",
            anchor_y="center",
            font_size=_scaled_font(12),
            color=(40, 200, 80, 255),
            batch=self.ui_batch,
        )
        self._sell_button_bounds = (
            sell_button_x,
            sell_button_y,
            upgrade_button_width,
            upgrade_button_height,
        )
        self._sell_hidden_pos = (sell_button_x, sell_button_y)
        self._sell_visible = False
        self._refresh_sell_button()

        self._game_over_overlay = pyglet.shapes.Rectangle(
            0,
            0,
            self._world_len(self.map_image.width),
            self._world_len(self.map_image.height),
            color=(0, 0, 0),
            batch=self.batch,
        )
        self._game_over_overlay.opacity = 0
        self._game_over_label = pyglet.text.Label(
            "GAME LOST",
            x=self._world_len(self.map_image.width / 2),
            y=self._world_len(self.map_image.height / 2),
            anchor_x="center",
            anchor_y="center",
            font_size=int(round(self._world_len(36 * TEXT_SCALE))),
            color=(220, 30, 30, 0),
            batch=self.batch,
        )
        self._game_over_label_colors = {
            "lost": (220, 30, 30),
            "won": (30, 220, 30),
        }
        self._replay_notice_label = pyglet.text.Label(
            "",
            x=0,
            y=0,
            anchor_x="center",
            anchor_y="center",
            font_size=_scaled_font(24),
            color=(240, 240, 240, 0),
            batch=self.ui_batch,
        )
        self._replay_notice_time = None
        self._update_replay_notice_position()

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

    def _refresh_auto_wave_button(self) -> None:
        if self._replay is not None:
            return
        if self._auto_wave_button is None or self._auto_wave_label is None:
            return
        if self.state.auto_level:
            self._auto_wave_button.color = (70, 90, 70)
            self._auto_wave_button.border_color = (120, 180, 120)
            self._auto_wave_label.color = (240, 255, 240, 255)
        else:
            self._auto_wave_button.color = (70, 70, 74)
            self._auto_wave_button.border_color = (120, 120, 125)
            self._auto_wave_label.color = (240, 240, 240, 255)

    def _refresh_replay_controls(self) -> None:
        if self._replay is None:
            return
        if self._next_wave_button is not None and self._next_wave_label is not None:
            if self._replay_finished():
                self._next_wave_button.color = (70, 70, 74)
                self._next_wave_button.border_color = (120, 120, 125)
                self._next_wave_label.color = (240, 240, 240, 255)
                self._next_wave_label.text = "Restart"
            elif self._replay_playing:
                self._next_wave_button.color = (70, 90, 70)
                self._next_wave_button.border_color = (120, 180, 120)
                self._next_wave_label.color = (240, 255, 240, 255)
                self._next_wave_label.text = "Pause"
            else:
                self._next_wave_button.color = (70, 70, 74)
                self._next_wave_button.border_color = (120, 120, 125)
                self._next_wave_label.color = (240, 240, 240, 255)
                self._next_wave_label.text = "Play"
        if self._auto_wave_button is not None and self._auto_wave_label is not None:
            self._auto_wave_label.text = "Pause at end"
            if self._replay_pause_at_end:
                self._auto_wave_button.color = (70, 90, 70)
                self._auto_wave_button.border_color = (120, 180, 120)
                self._auto_wave_label.color = (240, 255, 240, 255)
            else:
                self._auto_wave_button.color = (70, 70, 74)
                self._auto_wave_button.border_color = (120, 120, 125)
                self._auto_wave_label.color = (240, 240, 240, 255)

    def _refresh_wave_info(self) -> None:
        info = wave_display_info(self.state, self.map_data)
        current_num = self.state.level if info["current"] else None
        next_num = self.state.level + 1 if info["next"] else None
        self._apply_wave_row(self._wave_now_sprite, self._wave_now_label, info["current"], current_num)
        self._apply_wave_row(self._wave_next_sprite, self._wave_next_label, info["next"], next_num)

    def _apply_wave_row(self, sprite, label, info: dict | None, wave_number: int | None) -> None:
        if sprite is None or label is None:
            return
        if not info:
            sprite.opacity = 0
            label.text = ""
            return
        prefix = f"Wave {wave_number} : " if wave_number is not None else ""
        label.text = f"{prefix}{info['name']}\n{info['hp']} hp"
        show_sprite = bool(info.get("show_sprite", True))
        if not show_sprite:
            sprite.opacity = 0
            return
        creep_type = int(info.get("type", 1))
        img = self.creep_images.get(creep_type, self.creep_images[1])
        if sprite.image is not img:
            sprite.image = img
            if img.width > 0:
                sprite.scale = self._wave_icon_size / img.width
        sprite.opacity = 255

    def _can_afford_build(self, kind: str) -> bool:
        if kind in BUFF_UI_KINDS:
            return getattr(self.state, "ups", 0) >= 1
        tower_def = get_tower_def(kind)
        return self.state.bank >= tower_def.cost

    def _refresh_tower_buttons(self) -> None:
        if self._placement_active and self._placement_kind in ("bonus_interest", "bonus_lives"):
            self._placement_active = False
            self._tower_preview_sprite.opacity = 0
            self._placement_range_shape.opacity = 0
            self._placement_cell = None
            self._placement_valid = False
        if self._placement_active:
            if not self._can_afford_build(self._placement_kind):
                self._placement_active = False
                self._tower_preview_sprite.opacity = 0
                self._placement_range_shape.opacity = 0
                self._placement_cell = None
                self._placement_valid = False
        self._tower_button_enabled.clear()
        for button in self._tower_buttons:
            can_afford = self._can_afford_build(button.kind)
            self._tower_button_enabled[button.kind] = can_afford
            is_active = self._placement_active and self._placement_kind == button.kind
            if is_active:
                button.rect.color = (70, 90, 70)
                button.rect.border_color = (120, 180, 120)
                button.rect.opacity = 255
                if button.sprite is not None:
                    button.sprite.color = (255, 255, 255)
                if button.label is not None:
                    button.label.color = (240, 240, 240, 255)
                continue
            if can_afford:
                button.rect.color = (60, 60, 64)
                button.rect.border_color = (110, 110, 115)
                button.rect.opacity = 255
                if button.sprite is not None:
                    button.sprite.color = (255, 255, 255)
                if button.label is not None:
                    button.label.color = (240, 240, 240, 255)
            else:
                button.rect.color = (40, 40, 44)
                button.rect.border_color = (80, 80, 85)
                button.rect.opacity = 170
                if button.sprite is not None:
                    button.sprite.color = (130, 130, 130)
                if button.label is not None:
                    button.label.color = (130, 130, 130, 180)

    def _clear_target_mode_buttons(self) -> None:
        for button in self._target_mode_buttons:
            button.rect.delete()
            button.label.delete()
        self._target_mode_buttons.clear()

    def _build_target_mode_buttons(self, modes: tuple[str, ...]) -> None:
        self._clear_target_mode_buttons()
        if not modes:
            self._target_mode_button_modes = ()
            return
        row_x, row_y, row_w, row_h = self._target_mode_row_bounds
        button_spacing = 6
        count = len(modes)
        button_w = max(0.0, (row_w - button_spacing * (count - 1)) / count)
        button_h = max(0.0, min(26.0, row_h))
        button_y = row_y + (row_h - button_h) / 2 if row_h > button_h else row_y
        for index, mode in enumerate(modes):
            button_x = row_x + index * (button_w + button_spacing)
            rect = pyglet.shapes.BorderedRectangle(
                button_x,
                button_y,
                button_w,
                button_h,
                color=(60, 60, 64),
                border=2,
                border_color=(110, 110, 115),
                batch=self.ui_batch,
            )
            label = pyglet.text.Label(
                _target_mode_label(mode),
                x=button_x + button_w / 2,
                y=button_y + button_h / 2,
                anchor_x="center",
                anchor_y="center",
                font_size=_scaled_font(10),
                color=(230, 230, 230, 255),
                batch=self.ui_batch,
            )
            bounds = (button_x, button_y, button_w, button_h)
            self._target_mode_buttons.append(_TargetModeButton(mode, bounds, rect, label))
        self._target_mode_button_modes = tuple(modes)

    def _refresh_target_mode_buttons(self) -> None:
        tower = self._selected_tower
        if tower is None:
            if self._target_mode_buttons:
                self._clear_target_mode_buttons()
            self._target_mode_button_modes = ()
            return
        if tower.kind in ("buffD", "buffR"):
            if self._target_mode_buttons:
                self._clear_target_mode_buttons()
            self._target_mode_button_modes = ()
            return
        tower_def = get_tower_def(tower.kind)
        modes = tuple(tower_def.target_modes)
        if modes != self._target_mode_button_modes:
            self._build_target_mode_buttons(modes)
        active_mode = str(getattr(tower, "target_mode", "closest") or "closest")
        for button in self._target_mode_buttons:
            if button.mode == active_mode:
                button.rect.color = (70, 90, 70)
                button.rect.border_color = (120, 180, 120)
                button.label.color = (240, 255, 240, 255)
            else:
                button.rect.color = (60, 60, 64)
                button.rect.border_color = (110, 110, 115)
                button.label.color = (230, 230, 230, 255)

    def _target_mode_button_at(self, logical_x: float, logical_y: float) -> str | None:
        for button in self._target_mode_buttons:
            if _point_in_rect(logical_x, logical_y, button.bounds):
                return button.mode
        return None

    def _handle_target_mode_click(self, mode: str) -> None:
        if self._selected_tower is None:
            return
        if self.state.paused or self.state.game_over:
            return
        if not set_target_mode(self._selected_tower, mode):
            return
        self._refresh_target_mode_buttons()

    def _placement_preview_range(self, kind: str) -> float:
        if kind in ("buffD", "buffR"):
            return 100.0
        return float(get_tower_def(kind).range)

    def _set_placement_active(self, active: bool) -> None:
        if active == self._placement_active:
            return
        self._placement_active = active
        self._refresh_tower_buttons()
        if active:
            self._set_selected_tower(None)
        if not active:
            self._tower_preview_sprite.opacity = 0
            self._placement_range_shape.opacity = 0
            self._placement_cell = None
            self._placement_valid = False
            return
        if self._last_mouse_pos is not None:
            self._update_placement_preview(*self._last_mouse_pos)

    def _handle_next_wave_click(self) -> None:
        if self.state.paused:
            if self._replay is None:
                return
            self.state.paused = False
        if self._replay is not None:
            if self._replay_finished():
                self._restart_replay(start_playing=True)
                return
        if self.state.game_over:
            return
        if self._replay is not None:
            if self._replay_playing:
                self._replay_playing = False
                self._refresh_replay_controls()
                return
            self._replay_playing = True
            self._refresh_replay_controls()
            if not self._replay_in_wave:
                self._start_prepared_replay_wave()
            return
        self._auto_send_armed = True
        start_next_wave(self.state, self.map_data)

    def _toggle_auto_wave(self) -> None:
        if self._replay is not None:
            self._replay_pause_at_end = not self._replay_pause_at_end
            self._refresh_replay_controls()
            if self._replay_playing and not self._replay_pause_at_end and not self._replay_in_wave:
                self._start_prepared_replay_wave()
            return
        self.state.auto_level = not self.state.auto_level
        self._refresh_auto_wave_button()
        if self.state.auto_level and self._auto_send_armed:
            maybe_auto_next_wave(self.state, self.map_data)

    def _set_game_over_overlay(self, active: bool, *, test_result: str | None = None) -> None:
        if active:
            self._game_over_overlay.opacity = 150
            if test_result == "passed":
                text = "TEST PASSED"
                color = self._game_over_label_colors["won"]
            elif test_result == "failed":
                text = "TEST FAILED"
                color = self._game_over_label_colors["lost"]
            elif self.state.game_won:
                text = "GAME WON"
                color = self._game_over_label_colors["won"]
            else:
                text = "GAME LOST"
                color = self._game_over_label_colors["lost"]
            self._game_over_label.text = text
            self._game_over_label.color = (*color, 255)
        else:
            self._game_over_overlay.opacity = 0

    def _maybe_finish_game(self) -> None:
        if self.state.game_over:
            return
        if not self.state.creeps and self.state.level >= len(LEVELS):
            self.state.game_over = True
            self.state.game_won = True
            self._game_over_label.color = (*self._game_over_label_colors["lost"], 0)

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
        border_size = max(1, int(round(self._world_len(1))))
        grid_size = self._world_len(grid)
        for x, y in cells:
            draw_x = self._world_len(x + self._world_offset[0])
            draw_y = self._world_len(self.map_image.height - (y + grid + self._world_offset[1]))
            rect = pyglet.shapes.BorderedRectangle(
                draw_x,
                draw_y,
                grid_size,
                grid_size,
                color=fill_color,
                border=border_size,
                border_color=border_color,
                batch=self.batch,
            )
            rect.opacity = 120
            rect.border_opacity = 200
            self._grid_shapes.append(rect)

    def _tower_draw_position(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        grid = self.map_data.grid
        draw_x = self._world_len(cell_x + grid * 0.5 + self._world_offset[0])
        draw_y = self._world_len(self.map_image.height - (cell_y + grid * 0.5 + self._world_offset[1]))
        return draw_x, draw_y

    def _tower_level_tag_position(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        grid = self.map_data.grid
        tag_size = self._tower_level_tag_base
        draw_x = self._world_len(cell_x + self._world_offset[0] + 2)
        draw_y = self._world_len(
            self.map_image.height - (cell_y + grid + self._world_offset[1]) + grid - tag_size - 2
        )
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
            self._placement_range_shape.opacity = 0
            return
        cell = self._grid_cell_at(logical_x, logical_y)
        if cell is None:
            self._tower_preview_sprite.opacity = 0
            self._placement_range_shape.opacity = 0
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
        self._placement_range_shape.x = draw_x
        self._placement_range_shape.y = draw_y
        radius = self._placement_preview_range(self._placement_kind)
        self._placement_range_shape.radius = self._world_len(radius)
        self._placement_range_shape.color = (80, 200, 120) if valid else (200, 80, 80)
        self._placement_range_shape.opacity = 90
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

    def _tower_button_kind_at(self, logical_x: float, logical_y: float) -> str | None:
        for button in self._tower_buttons:
            if _point_in_rect(logical_x, logical_y, button.bounds):
                return button.kind
        return None

    def _handle_tower_selection(self, logical_x: float, logical_y: float) -> None:
        tower = self._tower_at(logical_x, logical_y)
        self._set_selected_tower(tower)

    def _set_selected_tower(self, tower) -> None:
        self._selected_tower = tower
        if tower is None:
            self._tower_range_shape.opacity = 0
            if self._hovered_tower_kind is None:
                self._set_info_text("", None)
            self._refresh_upgrade_button()
            self._refresh_sell_button()
            self._refresh_target_mode_buttons()
            return
        draw_x, draw_y = self._tower_draw_position(tower.cell_x, tower.cell_y)
        self._tower_range_shape.x = draw_x
        self._tower_range_shape.y = draw_y
        self._tower_range_shape.radius = self._world_len(tower.range)
        self._tower_range_shape.opacity = 120
        if self._hovered_tower_kind is None:
            self._set_info_text(self._tower_info_text_from_tower(tower), tower)
        self._refresh_upgrade_button()
        self._refresh_sell_button()
        self._refresh_target_mode_buttons()

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
        if kind == "buffD":
            return "Buff Damage\nCOST 1 bonus point\n+25% damage to towers in radius"
        if kind == "buffR":
            return "Buff Range\nCOST 1 bonus point\n+25% range to towers in radius"
        if kind == "bonus_interest":
            return "Interest Boost\nCOST 1 bonus point\nInterest +3% (applied at wave start)"
        if kind == "bonus_lives":
            return "Emergency Lives\nCOST 1 bonus point\nLives +5"
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
        hovered_kind = self._tower_button_kind_at(logical_x, logical_y)
        if hovered_kind == self._hovered_tower_kind:
            return
        self._hovered_tower_kind = hovered_kind
        if hovered_kind is not None:
            self._set_info_text(self._tower_info_text_from_kind(hovered_kind), None)
        else:
            if self._selected_tower is not None:
                self._set_info_text(
                    self._tower_info_text_from_tower(self._selected_tower),
                    self._selected_tower,
                )
            else:
                self._set_info_text("", None)

    def _format_bonus_value(self, value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) < 0.01:
            return str(int(rounded))
        return f"{value:.1f}".rstrip("0").rstrip(".")

    def _measure_info_text_width(self, text: str, font_size: int) -> float:
        label = self._tower_info_measure_label
        label.font_name = self._tower_info_label.font_name
        label.font_size = font_size
        label.text = text
        return label.content_width

    def _clear_info_bonus(self) -> None:
        self._tower_info_bonus_label.text = ""
        self._tower_info_bonus_label.color = (40, 200, 80, 0)

    def _apply_info_bonus(self, tower) -> None:
        if tower is None or tower.kind in ("buffD", "buffR"):
            self._clear_info_bonus()
            return
        buffed_damage = getattr(tower, "buffed_damage", None)
        buffed_range = getattr(tower, "buffed_range", None)
        if buffed_damage is None or buffed_range is None:
            self._clear_info_bonus()
            return
        dmg_bonus = buffed_damage - tower.damage
        range_bonus = buffed_range - tower.range
        if dmg_bonus <= 0.01 and range_bonus <= 0.01:
            self._clear_info_bonus()
            return
        dmg_text = f"+{self._format_bonus_value(dmg_bonus)}" if dmg_bonus > 0.01 else ""
        range_text = f"+{self._format_bonus_value(range_bonus)}m" if range_bonus > 0.01 else ""
        font_size = self._tower_info_label.font_size
        self._tower_info_bonus_label.font_size = font_size
        self._tower_info_bonus_label.text = f"\n{dmg_text}\n{range_text}"
        base_width = max(
            self._measure_info_text_width(f"DMG: {tower.damage}", font_size),
            self._measure_info_text_width(f"RANGE: {tower.range}m", font_size),
        )
        self._tower_info_bonus_label.x = self._tower_info_label.x + base_width + 6
        self._tower_info_bonus_label.y = self._tower_info_label.y
        self._tower_info_bonus_label.color = (40, 200, 80, 255)
        info_x, _, info_w, _ = self._tower_info_bounds
        max_x = info_x + info_w - 8 - self._tower_info_bonus_label.content_width
        if self._tower_info_bonus_label.x > max_x:
            self._tower_info_bonus_label.x = max(info_x + 8, max_x)

    def _set_info_text(self, text: str, tower) -> None:
        base_size = _scaled_font(10)
        min_size = max(1, _scaled_font(7))
        _, _, info_w, info_h = self._tower_info_bounds
        max_height = max(0, info_h - 16)
        max_width = max(0, info_w - 16)
        if max_height <= 0 or max_width <= 0:
            self._tower_info_label.text = ""
            self._clear_info_bonus()
            return
        self._tower_info_label.width = int(max_width)
        self._tower_info_bonus_label.width = int(max_width)
        size = base_size
        while True:
            self._tower_info_label.font_size = size
            self._tower_info_label.text = text
            if self._tower_info_label.content_height <= max_height or size <= min_size:
                break
            size -= 1
        if self._tower_info_label.content_height <= max_height:
            self._apply_info_bonus(tower)
            return
        self._tower_info_label.text = text
        self._tower_info_label.text = self._truncate_info_text(text, max_height)
        self._apply_info_bonus(tower)

    def _truncate_info_text(self, text: str, max_height: float) -> str:
        label = self._tower_info_label
        trimmed = text.rstrip()
        if not trimmed:
            return ""
        suffix = "..."
        while trimmed:
            trimmed = trimmed[:-1].rstrip()
            candidate = f"{trimmed}{suffix}" if trimmed else suffix
            label.text = candidate
            if label.content_height <= max_height:
                return candidate
        return ""

    def _position_button_price_label(
        self,
        base_label: pyglet.text.Label,
        price_label: pyglet.text.Label,
        bounds: tuple[float, float, float, float],
    ) -> None:
        price_label.y = base_label.y
        price_label.x = base_label.x + base_label.content_width + 6
        min_x = bounds[0] + 8
        max_x = bounds[0] + bounds[2] - 8 - price_label.content_width
        if price_label.x > max_x:
            price_label.x = max(min_x, max_x)

    def _refresh_upgrade_button(self) -> None:
        if self._selected_tower is None or self._selected_tower.level >= 10:
            self._upgrade_visible = False
        elif self._selected_tower.kind in ("buffD", "buffR"):
            self._upgrade_visible = False
        else:
            self._upgrade_visible = True
        alpha = 255 if self._upgrade_visible else 0
        self._upgrade_button.opacity = alpha
        self._upgrade_button.border_opacity = alpha
        if not self._upgrade_visible:
            self._upgrade_button.x = -1000
            self._upgrade_button.y = -1000
            self._upgrade_button_label.x = -1000
            self._upgrade_button_label.y = -1000
            self._upgrade_price_label.x = -1000
            self._upgrade_price_label.y = -1000
            self._upgrade_price_label.text = ""
            return
        self._upgrade_button.x, self._upgrade_button.y = self._upgrade_hidden_pos
        self._upgrade_button_label.x, self._upgrade_button_label.y = self._upgrade_label_pos
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
        upgrade_cost = int(self._selected_tower.base_cost / 2)
        can_afford = self.state.bank >= upgrade_cost
        self._upgrade_button.color = (60, 60, 64) if can_afford else (40, 40, 44)
        self._upgrade_button.border_color = (110, 110, 115) if can_afford else (80, 80, 85)
        self._upgrade_button_label.color = (230, 230, 230, 255 if can_afford else 160)
        price_color = (40, 200, 80, 255) if can_afford else (220, 60, 60, 255)
        self._upgrade_price_label.color = price_color
        self._upgrade_price_label.text = f"${upgrade_cost}"
        self._position_button_price_label(
            self._upgrade_button_label,
            self._upgrade_price_label,
            self._upgrade_button_bounds,
        )

    def _refresh_sell_button(self) -> None:
        if self._selected_tower is None:
            self._sell_visible = False
        elif self._selected_tower.kind in ("buffD", "buffR"):
            self._sell_visible = False
        else:
            self._sell_visible = True
        alpha = 255 if self._sell_visible else 0
        self._sell_button.opacity = alpha
        self._sell_button.border_opacity = alpha
        if not self._sell_visible:
            self._sell_button.x = -1000
            self._sell_button.y = -1000
            self._sell_button_label.x = -1000
            self._sell_button_label.y = -1000
            self._sell_price_label.x = -1000
            self._sell_price_label.y = -1000
            self._sell_price_label.text = ""
            return
        self._sell_button.x, self._sell_button.y = self._sell_hidden_pos
        self._sell_button_label.x, self._sell_button_label.y = self._sell_label_pos
        self._sell_button_label.color = (
            self._sell_button_label.color[0],
            self._sell_button_label.color[1],
            self._sell_button_label.color[2],
            alpha,
        )
        self._sell_price_label.color = (
            self._sell_price_label.color[0],
            self._sell_price_label.color[1],
            self._sell_price_label.color[2],
            alpha,
        )
        sell_price = int(self._selected_tower.cost / 100 * 75)
        self._sell_button.color = (60, 60, 64)
        self._sell_button.border_color = (110, 110, 115)
        self._sell_button_label.color = (230, 230, 230, 255)
        self._sell_price_label.color = (40, 200, 80, 255)
        self._sell_price_label.text = f"${sell_price}"
        self._position_button_price_label(
            self._sell_button_label,
            self._sell_price_label,
            self._sell_button_bounds,
        )

    def _handle_upgrade_click(self) -> None:
        if self._selected_tower is None:
            return
        upgraded = upgrade_tower(self.state, self._selected_tower)
        if not upgraded:
            return
        self._set_selected_tower(self._selected_tower)
        self._refresh_upgrade_button()
        self._refresh_sell_button()

    def _handle_sell_click(self) -> None:
        if self._selected_tower is None:
            return
        if sell_tower(self.state, self._selected_tower) is None:
            return
        self._set_selected_tower(None)
        self._sync_tower_sprites()
        self._sync_tower_shots()
        self._refresh_tower_buttons()

    def _update_view_transform(self) -> None:
        if self._logical_width <= 0 or self._logical_height <= 0:
            self._base_view_scale = 1.0
            self._base_view_offset_x = 0.0
            self._base_view_offset_y = 0.0
            self._ui_view_scale = 1.0
            self._ui_view_offset_x = 0.0
            self._ui_view_offset_y = 0.0
            self._world_view_scale = 1.0
            self._world_view_offset_x = 0.0
            self._world_view_offset_y = 0.0
            return
        self._ui_layout_height = self._logical_height / max(0.001, self._ui_scale)
        scale_x = self._window_width / self._logical_width
        scale_y = self._window_height / self._logical_height
        self._base_view_scale = min(scale_x, scale_y)
        self._base_view_offset_x = (self._window_width - self._logical_width * self._base_view_scale) / 2.0
        self._base_view_offset_y = (self._window_height - self._logical_height * self._base_view_scale) / 2.0
        self._world_view_scale = self._base_view_scale / max(0.001, self._world_render_scale)
        self._world_view_offset_x = self._base_view_offset_x
        self._world_view_offset_y = self._base_view_offset_y
        self._ui_view_scale = self._base_view_scale * self._ui_scale
        self._ui_view_offset_x = (
            self._base_view_offset_x
            + self._sidebar_x * self._base_view_scale
            - self._sidebar_x * self._ui_view_scale
        )
        self._ui_view_offset_y = (
            self._base_view_offset_y
            + self._logical_height * self._base_view_scale
            - self._ui_layout_height * self._ui_view_scale
        )

    def _to_logical_coords(self, x: float, y: float, *, scale: float, offset_x: float, offset_y: float) -> tuple[float, float]:
        fb_w, fb_h = self.window.get_framebuffer_size()
        if fb_w > 0 and fb_h > 0:
            x *= self._window_width / fb_w
            y *= self._window_height / fb_h
        return (x - offset_x) / scale, (y - offset_y) / scale

    def _to_world_coords(self, x: float, y: float) -> tuple[float, float]:
        return self._to_logical_coords(
            x,
            y,
            scale=self._base_view_scale,
            offset_x=self._base_view_offset_x,
            offset_y=self._base_view_offset_y,
        )

    def _to_ui_coords(self, x: float, y: float) -> tuple[float, float]:
        return self._to_logical_coords(
            x,
            y,
            scale=self._ui_view_scale,
            offset_x=self._ui_view_offset_x,
            offset_y=self._ui_view_offset_y,
        )

    def _world_len(self, value: float) -> float:
        return value * self._world_render_scale

    def _world_draw_x(self, world_x: float) -> float:
        return self._world_len(world_x + self._world_offset[0])

    def _world_draw_y(self, world_y: float) -> float:
        return self._world_len(self.map_image.height - (world_y + self._world_offset[1]))

    def _capture_worker(self) -> None:
        while True:
            tick, image = self._capture_tasks.get()
            filename = self._screenshot_dir / f"tick_{tick:06d}.png"
            try:
                image.save(str(filename))
            except Exception as exc:
                print(f"[gui] screenshot save failed tick={tick}: {exc}")
            finally:
                self._capture_tasks.task_done()


def _point_in_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= (rx + rw) and ry <= y <= (ry + rh)


def run(
    map_path: str | Path = Path("data/maps/switchback.json"),
    *,
    test_definition: TestDefinition | None = None,
    test_speed: float = TEST_MODE_SPEED,
    replay: Replay | None = None,
    replay_path: Path | None = None,
) -> None:
    run_dir = _next_run_dir(Path("runs") / "gui")
    log_path = run_dir / "log.txt"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    _app = SimpleGui(
        Path(map_path),
        run_dir,
        test_definition=test_definition,
        test_speed=test_speed,
        replay=replay,
        replay_path=replay_path,
    )
    pyglet.app.run()


def _resolve_replay_map_path(map_path: str | Path) -> Path:
    p = Path(map_path)
    if p.is_absolute():
        return p
    root = Path(__file__).resolve().parents[3]
    if p.suffix:
        return root / p
    if p.parent == Path("."):
        return root / "data/maps" / f"{p.name}.json"
    return root / p.with_suffix(".json")


def run_replay(replay_path: str | Path) -> None:
    replay = load_replay(replay_path)
    map_path = _resolve_replay_map_path(replay.map_path)
    run(map_path, replay=replay, replay_path=Path(replay_path))
