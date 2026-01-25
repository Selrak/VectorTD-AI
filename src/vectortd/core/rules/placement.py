from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re
import struct
import zlib
from typing import Iterable

from ..model.entities import Tower
from ..model.towers import BUFF_TOWER_UPS_COST, get_tower_def, is_buff_tower
from .buffs import recompute_buffs


_ALLOWED_CELLS_SET_CACHE: dict[int, set[tuple[int, int]]] = {}


@dataclass(frozen=True)
class HitMask:
    width: int
    height: int
    offset_x: float
    offset_y: float
    alpha_rows: tuple[bytes, ...]

    def hit_test(self, x: float, y: float) -> bool:
        px = int(x - self.offset_x)
        py = int(y - self.offset_y)
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        return self.alpha_rows[py][px] > 0


def buildable_cells(map_data, *, path_half_width: float | None = None) -> list[tuple[int, int]]:
    grid = int(getattr(map_data, "grid", 25))
    width = int(getattr(map_data, "width", 0))
    height = int(getattr(map_data, "height", 0))
    if width <= 0 or height <= 0 or grid <= 0:
        return []

    allowed_cells = _allowed_cells_from_map(map_data)
    if allowed_cells is not None:
        return list(allowed_cells)

    hit_mask = _hit_mask_for_map(map_data)
    if hit_mask is not None:
        return _buildable_cells_from_mask(width, height, grid, hit_mask)

    segments = _path_segments(map_data)
    half_width = float(path_half_width if path_half_width is not None else grid)
    threshold_sq = half_width * half_width
    cells: list[tuple[int, int]] = []
    for y in range(0, height, grid):
        for x in range(0, width, grid):
            cx = x + grid * 0.5
            cy = y + grid * 0.5
            if _point_hits_path(cx, cy, segments, threshold_sq):
                continue
            cells.append((x, y))
    return cells


def cell_is_buildable(
    map_data,
    cell_x: int,
    cell_y: int,
    *,
    path_half_width: float | None = None,
) -> bool:
    grid = int(getattr(map_data, "grid", 25))
    cx = cell_x + grid * 0.5
    cy = cell_y + grid * 0.5
    allowed_cells = _allowed_cells_set(map_data)
    if allowed_cells is not None:
        return (cell_x, cell_y) in allowed_cells

    hit_mask = _hit_mask_for_map(map_data)
    if hit_mask is not None:
        return not hit_mask.hit_test(cx, cy)

    segments = _path_segments(map_data)
    half_width = float(path_half_width if path_half_width is not None else grid)
    threshold_sq = half_width * half_width
    return not _point_hits_path(cx, cy, segments, threshold_sq)


def can_place_tower(
    state,
    map_data,
    cell_x: int,
    cell_y: int,
    tower_kind: str = "green",
    *,
    path_half_width: float | None = None,
) -> bool:
    grid = int(getattr(map_data, "grid", 25))
    width = int(getattr(map_data, "width", 0))
    height = int(getattr(map_data, "height", 0))
    if grid <= 0 or width <= 0 or height <= 0:
        return False
    if cell_x < 0 or cell_y < 0 or cell_x + grid > width or cell_y + grid > height:
        return False
    if not cell_is_buildable(map_data, cell_x, cell_y, path_half_width=path_half_width):
        return False
    bank = getattr(state, "bank", None)
    tower_def = get_tower_def(str(tower_kind))
    if is_buff_tower(tower_def.kind):
        ups = getattr(state, "ups", None)
        if ups is None or int(ups) < BUFF_TOWER_UPS_COST:
            return False
    elif bank is not None:
        if int(bank) < tower_def.cost:
            return False
    for tower in getattr(state, "towers", []):
        if getattr(tower, "cell_x", None) == cell_x and getattr(tower, "cell_y", None) == cell_y:
            return False
    return True


def place_tower(
    state,
    map_data,
    cell_x: int,
    cell_y: int,
    tower_kind: str = "green",
    *,
    path_half_width: float | None = None,
) -> Tower | None:
    if not can_place_tower(
        state,
        map_data,
        cell_x,
        cell_y,
        tower_kind,
        path_half_width=path_half_width,
    ):
        return None
    towers = getattr(state, "towers", None)
    if towers is None:
        return None
    tower_def = get_tower_def(str(tower_kind))
    if is_buff_tower(tower_def.kind):
        ups = getattr(state, "ups", None)
        if ups is not None:
            state.ups = int(ups) - BUFF_TOWER_UPS_COST
    else:
        bank = getattr(state, "bank", None)
        if bank is not None:
            state.bank = int(bank) - tower_def.cost
    tower = Tower(
        cell_x=int(cell_x),
        cell_y=int(cell_y),
        kind=tower_def.kind,
        title=tower_def.title,
        level=tower_def.level,
        cost=tower_def.cost,
        range=tower_def.range,
        damage=tower_def.damage,
        description=tower_def.description,
        base_cost=tower_def.base_cost,
        base_range=tower_def.base_range,
        base_damage=tower_def.base_damage,
        target_mode=tower_def.target_mode,
        rof=tower_def.rof,
        cooldown=0.0,
        retarget_delay=(10 if tower_def.rof == 0 else tower_def.rof),
        target=None,
        shot_timer=0.0,
        shot_x=0.0,
        shot_y=0.0,
        shot_segments=[],
        shot_opacity=200,
    )
    towers.append(tower)
    recompute_buffs(state, map_data)
    return tower


def upgrade_tower(state, tower, map_data=None) -> bool:
    if tower is None:
        return False
    if is_buff_tower(getattr(tower, "kind", "")):
        return False
    if tower.level >= 10:
        return False
    upgrade_cost = int(tower.base_cost / 2)
    bank = getattr(state, "bank", None)
    if bank is None or int(bank) < upgrade_cost:
        return False
    tower.level += 1
    tower.damage += int(tower.base_damage / 2.2)
    tower.range += int(tower.base_range / 20)
    tower.cost += upgrade_cost
    state.bank = int(bank) - upgrade_cost
    if map_data is not None:
        recompute_buffs(state, map_data)
    return True


def sell_tower(state, tower, map_data=None) -> int | None:
    if tower is None:
        return None
    if is_buff_tower(getattr(tower, "kind", "")):
        return None
    towers = getattr(state, "towers", None)
    if towers is None or tower not in towers:
        return None
    sale_price = int(tower.cost / 100 * 75)
    bank = getattr(state, "bank", None)
    if bank is not None:
        state.bank = int(bank) + sale_price
    towers.remove(tower)
    pulses = getattr(state, "pulses", None)
    if pulses:
        state.pulses = [pulse for pulse in pulses if pulse.tower is not tower]
    if map_data is not None:
        recompute_buffs(state, map_data)
    return sale_price


def set_target_mode(tower, mode: str) -> bool:
    if tower is None:
        return False
    try:
        tower_def = get_tower_def(str(tower.kind))
    except KeyError:
        return False
    target_mode = str(mode)
    if target_mode not in tower_def.target_modes:
        return False
    tower.target_mode = target_mode
    tower.target = None
    return True


def _allowed_cells_from_map(map_data) -> list[tuple[int, int]] | None:
    cells = getattr(map_data, "buildable_cells", None)
    if cells:
        return list(cells)
    return None


def _allowed_cells_set(map_data) -> set[tuple[int, int]] | None:
    cells = getattr(map_data, "buildable_cells", None)
    if not cells:
        return None
    key = id(map_data)
    cached = _ALLOWED_CELLS_SET_CACHE.get(key)
    if cached is not None:
        return cached
    cell_set = set((int(cell[0]), int(cell[1])) for cell in cells)
    _ALLOWED_CELLS_SET_CACHE[key] = cell_set
    return cell_set


def _buildable_cells_from_mask(
    width: int,
    height: int,
    grid: int,
    hit_mask: HitMask,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for y in range(0, height, grid):
        for x in range(0, width, grid):
            cx = x + grid * 0.5
            cy = y + grid * 0.5
            if hit_mask.hit_test(cx, cy):
                continue
            cells.append((x, y))
    return cells


def _path_segments(map_data) -> list[tuple[float, float, float, float]]:
    segments: list[tuple[float, float, float, float]] = []
    for path in getattr(map_data, "paths", []):
        for a, b in zip(path[:-1], path[1:]):
            try:
                x1, y1 = map_data.marker_xy(a)
                x2, y2 = map_data.marker_xy(b)
            except Exception:
                continue
            segments.append((float(x1), float(y1), float(x2), float(y2)))
    return segments


def _point_hits_path(
    px: float,
    py: float,
    segments: Iterable[tuple[float, float, float, float]],
    threshold_sq: float,
) -> bool:
    for x1, y1, x2, y2 in segments:
        if _point_segment_distance_sq(px, py, x1, y1, x2, y2) <= threshold_sq:
            return True
    return False


def _point_segment_distance_sq(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _slugify_map_name(map_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", map_name.strip().lower())
    return slug.strip("_")


@lru_cache(maxsize=1)
def _mask_offsets() -> dict[str, tuple[float, float]]:
    path = _project_root() / "graphics" / "masks" / "offsets.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    offsets: dict[str, tuple[float, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            offsets[str(key)] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            continue
    return offsets


@lru_cache(maxsize=None)
def _load_hit_mask(path_str: str, offset_x: float, offset_y: float) -> HitMask | None:
    path = Path(path_str)
    if not path.exists():
        return None
    width, height, alpha_rows = _load_png_alpha(path)
    return HitMask(width, height, offset_x, offset_y, alpha_rows)


def _hit_mask_for_map(map_data) -> HitMask | None:
    name = getattr(map_data, "name", "")
    slug = _slugify_map_name(str(name))
    if not slug:
        return None
    path = _project_root() / "graphics" / "masks" / f"{slug}.png"
    if not path.exists():
        return None
    offsets = _mask_offsets()
    offset_x, offset_y = offsets.get(slug, (0.0, 0.0))
    return _load_hit_mask(str(path), offset_x, offset_y)


def _load_png_alpha(path: Path) -> tuple[int, int, tuple[bytes, ...]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    pos = 8
    width = height = None
    bit_depth = None
    color_type = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        pos += 8 + length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"Invalid PNG header: {path}")
    if bit_depth != 8 or color_type != 6:
        raise ValueError(f"Unsupported PNG format: {path}")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    rows: list[bytes] = []
    idx = 0
    prev = bytearray(stride)
    for _ in range(height):
        filter_type = raw[idx]
        idx += 1
        row = bytearray(raw[idx:idx + stride])
        idx += stride
        if filter_type == 1:
            for x in range(4, stride):
                row[x] = (row[x] + row[x - 4]) & 0xFF
        elif filter_type == 2:
            for x in range(stride):
                row[x] = (row[x] + prev[x]) & 0xFF
        elif filter_type == 3:
            for x in range(stride):
                left = row[x - 4] if x >= 4 else 0
                up = prev[x]
                row[x] = (row[x] + ((left + up) >> 1)) & 0xFF
        elif filter_type == 4:
            for x in range(stride):
                a = row[x - 4] if x >= 4 else 0
                b = prev[x]
                c = prev[x - 4] if x >= 4 else 0
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                row[x] = (row[x] + pr) & 0xFF
        alpha = bytes(row[3::4])
        rows.append(alpha)
        prev = row
    return width, height, tuple(rows)
