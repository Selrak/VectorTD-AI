from __future__ import annotations
from pathlib import Path
import argparse

from vectortd.core.model.map import load_map_json
from vectortd.core.model.state import GameState
from vectortd.core.rules.wave_spawner import dev_spawn_wave
from vectortd.core.rules.creep_motion import step_creeps


def _resolve_map_path(map_arg: str) -> Path:
    p = Path(map_arg)
    if p.suffix:
        return p
    if p.parent == Path("."):
        return Path("data/maps") / f"{p.name}.json"
    return p.with_suffix(".json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Map name (e.g. dev_2lanes) or path to json")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    map_data = load_map_json(_resolve_map_path(args.map))
    state = GameState()

    dev_spawn_wave(state, map_data)

    ticks = int(args.seconds * args.fps)
    for _ in range(ticks):
        step_creeps(state, map_data, dt_scale=1.0)
        if state.game_over:
            break

    # Petit résumé lisible
    min_pp = min((c.path_point for c in state.creeps), default=None)
    max_pp = max((c.path_point for c in state.creeps), default=None)
    print(f"creeps={len(state.creeps)} lives={state.lives} score={state.score} path_point=[{min_pp},{max_pp}]")
    if state.creeps:
        c0 = state.creeps[0]
        print(f"first_creep x={c0.x:.1f} y={c0.y:.1f} targ=({c0.targ_x:.1f},{c0.targ_y:.1f}) pp={c0.path_point}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
