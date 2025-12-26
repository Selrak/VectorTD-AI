from __future__ import annotations

from pathlib import Path

from vectortd.gui.pyglet_app import run


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    run(root / "data/maps/switchback.json")


if __name__ == "__main__":
    main()
