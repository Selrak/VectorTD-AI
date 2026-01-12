from __future__ import annotations

from pathlib import Path
import re
from typing import Dict


TOWER_IMAGES = {
    "green": "green.png",
    "green2": "green2.png",
    "green3": "green3.png",
    "red_refractor": "red_refractor.png",
    "red_spammer": "red_spammer.png",
    "red_rockets": "red_rockets.png",
    "purple1": "purple1.png",
    "purple2": "purple_2.png",
    "purple3": "purple3.png",
    "blue1": "blue1.png",
    "blue2": "blue2.png",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _slugify_map_name(map_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", map_name.strip().lower())
    return slug.strip("_")


def map_image_path(map_name: str) -> Path:
    slug = _slugify_map_name(map_name)
    if not slug:
        raise KeyError(f"Unknown map name: {map_name!r}")
    path = _project_root() / "graphics" / "maps" / f"{slug}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing map image: {path}")
    return path


def load_map_image(map_name: str):
    import pyglet

    path = map_image_path(map_name)
    return pyglet.image.load(str(path))


def load_creep_images() -> Dict[int, "pyglet.image.AbstractImage"]:
    import pyglet

    base = _project_root() / "graphics" / "creeps"
    images: Dict[int, "pyglet.image.AbstractImage"] = {}
    for i in range(1, 8):
        path = base / f"{i}.png"
        img = pyglet.image.load(str(path))
        img.anchor_x = img.width // 2
        img.anchor_y = img.height // 2
        images[i] = img
    return images


def tower_image_path(tower_kind: str) -> Path:
    filename = TOWER_IMAGES.get(tower_kind)
    if filename is None:
        raise KeyError(f"Unknown tower kind: {tower_kind!r}")
    path = _project_root() / "graphics" / "towers" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing tower image: {path}")
    return path


def load_tower_images() -> Dict[str, "pyglet.image.AbstractImage"]:
    import pyglet

    images: Dict[str, "pyglet.image.AbstractImage"] = {}
    for kind in TOWER_IMAGES:
        path = tower_image_path(kind)
        img = pyglet.image.load(str(path))
        img.anchor_x = img.width // 2
        img.anchor_y = img.height // 2
        images[kind] = img
    return images
