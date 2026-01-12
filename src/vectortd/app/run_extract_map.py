from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import re
import xml.etree.ElementTree as ET


_MARKER_RE = re.compile(r"^m(\d+)$")


def _parse_paths_and_spawn(script_path: Path) -> tuple[list[list[int]], str | None]:
    text = script_path.read_text(encoding="utf-8")
    paths_match = re.search(r"paths\s*=\s*(\[[^;]+)\s*;", text, re.S)
    if not paths_match:
        raise ValueError(f"Missing paths in {script_path}")
    paths_raw = ast.literal_eval(paths_match.group(1))
    if not isinstance(paths_raw, list):
        raise ValueError(f"Invalid paths in {script_path}")
    paths: list[list[int]] = []
    for path in paths_raw:
        if not isinstance(path, list):
            raise ValueError(f"Invalid path list in {script_path}")
        paths.append([int(x) for x in path])

    spawn_match = re.search(r'spawnDir\s*=\s*"([^"]+)"', text)
    spawn_dir = spawn_match.group(1) if spawn_match else None
    return paths, spawn_dir


def _parse_frame_scripts(scripts_dir: Path) -> dict[int, tuple[list[list[int]], str | None]]:
    frames: dict[int, tuple[list[list[int]], str | None]] = {}
    for frame_dir in sorted(scripts_dir.glob("frame_*")):
        match = re.match(r"frame_(\d+)", frame_dir.name)
        if not match:
            continue
        script_path = frame_dir / "DoAction.as"
        if not script_path.exists():
            continue
        frame_idx = int(match.group(1))
        frames[frame_idx] = _parse_paths_and_spawn(script_path)
    return frames


def _find_sprite_tag(root: ET.Element, sprite_id: int) -> ET.Element:
    target = str(sprite_id)
    for item in root.iter("item"):
        if item.get("type") == "DefineSpriteTag" and item.get("spriteId") == target:
            return item
    raise ValueError(f"Sprite {sprite_id} not found in XML")


def _matrix_translate(tag: ET.Element) -> tuple[float | None, float | None]:
    matrix = tag.find("matrix")
    if matrix is None:
        return None, None
    tx = matrix.get("translateX")
    ty = matrix.get("translateY")
    if tx is None or ty is None:
        return None, None
    try:
        return float(tx) / 20.0, float(ty) / 20.0
    except ValueError:
        return None, None


def _parse_sprite_frames(xml_path: Path, sprite_id: int) -> list[dict[int, tuple[float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sprite_tag = _find_sprite_tag(root, sprite_id)
    sub_tags = sprite_tag.find("subTags")
    if sub_tags is None:
        raise ValueError(f"Sprite {sprite_id} has no subTags")

    frames: list[dict[int, tuple[float, float]]] = []
    depth_state: dict[int, dict[str, float | str | None]] = {}

    for tag in list(sub_tags):
        if tag.tag != "item":
            continue
        tag_type = tag.get("type")
        if tag_type in {"PlaceObject2Tag", "PlaceObject3Tag"}:
            depth_attr = tag.get("depth")
            if depth_attr is None:
                continue
            depth = int(depth_attr)
            place_move = tag.get("placeFlagMove") == "true"
            name = tag.get("name")
            x, y = _matrix_translate(tag)
            if place_move:
                entry = depth_state.get(depth, {})
                if name:
                    entry["name"] = name
                if x is not None and y is not None:
                    entry["x"] = x
                    entry["y"] = y
                if entry:
                    depth_state[depth] = entry
            else:
                depth_state[depth] = {"name": name, "x": x, "y": y}
        elif tag_type in {"RemoveObject2Tag", "RemoveObjectTag"}:
            depth_attr = tag.get("depth")
            if depth_attr is None:
                continue
            depth_state.pop(int(depth_attr), None)
        elif tag_type == "ShowFrameTag":
            markers: dict[int, tuple[float, float]] = {}
            for entry in depth_state.values():
                name = entry.get("name")
                if not isinstance(name, str):
                    continue
                match = _MARKER_RE.match(name)
                if not match:
                    continue
                x = entry.get("x")
                y = entry.get("y")
                if x is None or y is None:
                    continue
                markers[int(match.group(1))] = (float(x), float(y))
            frames.append(markers)
    return frames


def _load_map_paths(maps_dir: Path) -> dict[int, Path]:
    map_paths: dict[int, Path] = {}
    for path in maps_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        level_index = int(data.get("level_index", 0))
        if level_index <= 0:
            continue
        if level_index in map_paths:
            raise ValueError(f"Duplicate level_index {level_index} in {path}")
        map_paths[level_index] = path
    return map_paths


def _update_map_data(
    data: dict,
    markers: dict[int, tuple[float, float]],
    paths: list[list[int]] | None,
    spawn_dir: str | None,
) -> tuple[dict, int, int]:
    if paths is not None:
        data["paths"] = paths
    if spawn_dir:
        data["spawn_dir"] = spawn_dir

    existing_markers = data.get("markers") or {}
    marker_ids: set[int] = set()
    for key in existing_markers.keys():
        try:
            marker_ids.add(int(key))
        except ValueError:
            continue
    for marker_id in markers.keys():
        marker_ids.add(int(marker_id))
    for path in data.get("paths", []):
        for marker_id in path:
            marker_ids.add(int(marker_id))

    merged: dict[str, list[float] | None] = {}
    for marker_id in sorted(marker_ids):
        if marker_id in markers:
            x, y = markers[marker_id]
            merged[str(marker_id)] = [float(x), float(y)]
        else:
            existing = existing_markers.get(str(marker_id))
            merged[str(marker_id)] = existing if existing is not None else None
    data["markers"] = merged
    missing = sum(1 for value in merged.values() if value is None)
    return data, len(merged), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="export/xml/swf.xml")
    ap.add_argument("--scripts-dir", default="export/scripts/DefineSprite_325")
    ap.add_argument("--maps-dir", default="data/maps")
    ap.add_argument("--sprite-id", type=int, default=325)
    ap.add_argument(
        "--level",
        type=int,
        action="append",
        default=[],
        help="Only update these level_index values (repeatable).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    scripts_dir = Path(args.scripts_dir)
    maps_dir = Path(args.maps_dir)

    frames = _parse_sprite_frames(xml_path, args.sprite_id)
    frame_scripts = _parse_frame_scripts(scripts_dir)
    map_paths = _load_map_paths(maps_dir)

    for level_index in sorted(map_paths.keys()):
        if args.level and level_index not in set(args.level):
            continue
        if level_index - 1 >= len(frames):
            print(f"level={level_index} skipped (no frame data)")
            continue
        markers = frames[level_index - 1]
        paths, spawn_dir = frame_scripts.get(level_index, (None, None))
        map_path = map_paths[level_index]
        data = json.loads(map_path.read_text(encoding="utf-8"))
        data, total, missing = _update_map_data(data, markers, paths, spawn_dir)
        if not args.dry_run:
            map_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"level={level_index} file={map_path} markers={total} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
