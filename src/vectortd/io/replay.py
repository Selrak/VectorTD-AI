from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
import logging
from typing import Any

from vectortd.core.engine import Engine
from vectortd.core.model.map import load_map_json
from vectortd.core.rng import seed_state

from vectortd.ai.actions import (
    Action,
    Noop,
    Place,
    Sell,
    SetMode,
    StartWave,
    Upgrade,
    action_from_dict,
    action_space_spec,
    action_to_dict,
    get_tower_slots,
    unflatten,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Replay:
    map_path: str
    seed: int
    waves: list[list[Action]]
    state_hashes: list[dict[str, Any]] | None = None
    final_summary: dict[str, Any] | None = None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _map_snapshot(map_data) -> dict[str, Any] | None:
    if map_data is None:
        return None
    markers = getattr(map_data, "markers", {}) or {}
    marker_items: list[tuple[int, float, float]] = []
    for key, coords in markers.items():
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            continue
        marker_items.append((_as_int(key), _as_float(coords[0]), _as_float(coords[1])))
    marker_items.sort(key=lambda item: item[0])
    paths_raw = getattr(map_data, "paths", []) or []
    paths = [[_as_int(point) for point in (path or [])] for path in paths_raw]
    return {
        "name": str(getattr(map_data, "name", "")),
        "level_index": _as_int(getattr(map_data, "level_index", 0)),
        "width": _as_int(getattr(map_data, "width", 0)),
        "height": _as_int(getattr(map_data, "height", 0)),
        "grid": _as_int(getattr(map_data, "grid", 0)),
        "spawn_dir": str(getattr(map_data, "spawn_dir", "")),
        "paths": paths,
        "markers": marker_items,
    }


def _slot_trace(state, max_towers: int) -> list[dict[str, Any] | None]:
    slots = get_tower_slots(state, max_towers)
    trace: list[dict[str, Any] | None] = []
    for tower in slots:
        if tower is None:
            trace.append(None)
            continue
        trace.append(
            {
                "kind": str(getattr(tower, "kind", "")),
                "level": _as_int(getattr(tower, "level", 0)),
                "cell_x": _as_int(getattr(tower, "cell_x", 0)),
                "cell_y": _as_int(getattr(tower, "cell_y", 0)),
            }
        )
    return trace


def _state_snapshot(state, map_data, spec, *, wave_ticks: int) -> dict[str, Any]:
    creeps = list(getattr(state, "creeps", []) or [])
    towers = list(getattr(state, "towers", []) or [])
    pulses = list(getattr(state, "pulses", []) or [])
    rockets = list(getattr(state, "rockets", []) or [])

    creep_index = {id(creep): idx for idx, creep in enumerate(creeps)}
    tower_index = {id(tower): idx for idx, tower in enumerate(towers)}
    max_towers = _as_int(getattr(spec, "max_towers", len(towers)))
    if max_towers <= 0:
        max_towers = len(towers)
    slot_trace = _slot_trace(state, max_towers)

    creep_snapshots = [
        {
            "x": _as_float(getattr(creep, "x", 0.0)),
            "y": _as_float(getattr(creep, "y", 0.0)),
            "type_id": _as_int(getattr(creep, "type_id", 0)),
            "path": [_as_int(p) for p in (getattr(creep, "path", []) or [])],
            "path_point": _as_int(getattr(creep, "path_point", 0)),
            "targ_x": _as_float(getattr(creep, "targ_x", 0.0)),
            "targ_y": _as_float(getattr(creep, "targ_y", 0.0)),
            "xval": _as_float(getattr(creep, "xval", 0.0)),
            "yval": _as_float(getattr(creep, "yval", 0.0)),
            "worth": _as_int(getattr(creep, "worth", 0)),
            "speed": _as_float(getattr(creep, "speed", 0.0)),
            "max_speed": _as_float(getattr(creep, "max_speed", 0.0)),
            "hp": _as_float(getattr(creep, "hp", 0.0)),
            "maxhp": _as_float(getattr(creep, "maxhp", 0.0)),
        }
        for creep in creeps
    ]

    tower_snapshots = [
        {
            "cell_x": _as_int(getattr(tower, "cell_x", 0)),
            "cell_y": _as_int(getattr(tower, "cell_y", 0)),
            "kind": str(getattr(tower, "kind", "")),
            "title": str(getattr(tower, "title", "")),
            "level": _as_int(getattr(tower, "level", 0)),
            "cost": _as_int(getattr(tower, "cost", 0)),
            "range": _as_int(getattr(tower, "range", 0)),
            "damage": _as_int(getattr(tower, "damage", 0)),
            "description": str(getattr(tower, "description", "")),
            "base_cost": _as_int(getattr(tower, "base_cost", 0)),
            "base_range": _as_int(getattr(tower, "base_range", 0)),
            "base_damage": _as_int(getattr(tower, "base_damage", 0)),
            "target_mode": str(getattr(tower, "target_mode", "")),
            "rof": _as_int(getattr(tower, "rof", 0)),
            "cooldown": _as_float(getattr(tower, "cooldown", 0.0)),
            "retarget_delay": _as_int(getattr(tower, "retarget_delay", 0)),
            "target_idx": creep_index.get(id(getattr(tower, "target", None)), -1)
            if getattr(tower, "target", None) is not None
            else -1,
            "shot_timer": _as_float(getattr(tower, "shot_timer", 0.0)),
            "shot_x": _as_float(getattr(tower, "shot_x", 0.0)),
            "shot_y": _as_float(getattr(tower, "shot_y", 0.0)),
            "shot_segments": [
                [
                    _as_float(seg[0]),
                    _as_float(seg[1]),
                    _as_float(seg[2]),
                    _as_float(seg[3]),
                ]
                for seg in (getattr(tower, "shot_segments", []) or [])
            ],
            "shot_opacity": _as_int(getattr(tower, "shot_opacity", 0)),
        }
        for tower in towers
    ]

    pulse_snapshots = [
        {
            "tower_idx": tower_index.get(id(getattr(pulse, "tower", None)), -1),
            "target_idx": creep_index.get(id(getattr(pulse, "target", None)), -1),
            "from_x": _as_float(getattr(pulse, "from_x", 0.0)),
            "from_y": _as_float(getattr(pulse, "from_y", 0.0)),
            "damage": _as_float(getattr(pulse, "damage", 0.0)),
            "alpha": _as_float(getattr(pulse, "alpha", 0.0)),
            "slow": bool(getattr(pulse, "slow", False)),
        }
        for pulse in pulses
    ]

    rocket_snapshots = [
        {
            "kind": str(getattr(rocket, "kind", "")),
            "x": _as_float(getattr(rocket, "x", 0.0)),
            "y": _as_float(getattr(rocket, "y", 0.0)),
            "target_idx": creep_index.get(id(getattr(rocket, "target", None)), -1)
            if getattr(rocket, "target", None) is not None
            else -1,
            "speed": _as_float(getattr(rocket, "speed", 0.0)),
            "damage": _as_float(getattr(rocket, "damage", 0.0)),
        }
        for rocket in rockets
    ]

    state_snapshot = {
        "bank": _as_int(getattr(state, "bank", 0)),
        "interest": _as_int(getattr(state, "interest", 0)),
        "ups": _as_int(getattr(state, "ups", 0)),
        "level": _as_int(getattr(state, "level", 0)),
        "lives": _as_int(getattr(state, "lives", 0)),
        "score": _as_int(getattr(state, "score", 0)),
        "bonus_every": _as_int(getattr(state, "bonus_every", 0)),
        "paused": bool(getattr(state, "paused", False)),
        "auto_level": bool(getattr(state, "auto_level", False)),
        "base_worth": _as_int(getattr(state, "base_worth", 0)),
        "base_hp": _as_int(getattr(state, "base_hp", 0)),
        "cc": _as_int(getattr(state, "cc", 0)),
        "rng_state": _as_int(getattr(state, "rng_state", 0)),
        "rng_calls": _as_int(getattr(state, "rng_calls", 0)),
        "last_wave_type": getattr(state, "last_wave_type", None),
        "last_wave_hp": _as_int(getattr(state, "last_wave_hp", 0)),
        "game_over": bool(getattr(state, "game_over", False)),
        "game_won": bool(getattr(state, "game_won", False)),
    }

    return {
        "map": _map_snapshot(map_data),
        "state": state_snapshot,
        "creeps": creep_snapshots,
        "towers": tower_snapshots,
        "pulses": pulse_snapshots,
        "rockets": rocket_snapshots,
        "slot_trace": slot_trace,
        "wave_ticks": _as_int(wave_ticks),
    }


def build_state_check(state, map_data, spec, *, wave_ticks: int) -> dict[str, Any]:
    snapshot = _state_snapshot(state, map_data, spec, wave_ticks=wave_ticks)
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _hash_payload(payload: Any) -> str:
        encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded_payload.encode("utf-8")).hexdigest()

    def _tower_sort_key(item: dict[str, Any]) -> tuple:
        return (
            _as_int(item.get("cell_y", 0)),
            _as_int(item.get("cell_x", 0)),
            str(item.get("kind", "")),
            _as_int(item.get("level", 0)),
            str(item.get("target_mode", "")),
        )

    def _creep_sort_key(item: dict[str, Any]) -> tuple:
        return (
            _as_int(item.get("type_id", 0)),
            _as_int(item.get("path_point", 0)),
            _as_float(item.get("x", 0.0)),
            _as_float(item.get("y", 0.0)),
            _as_float(item.get("hp", 0.0)),
            _as_float(item.get("maxhp", 0.0)),
        )

    def _pulse_sort_key(item: dict[str, Any]) -> tuple:
        return (
            _as_int(item.get("tower_idx", -1)),
            _as_int(item.get("target_idx", -1)),
            _as_float(item.get("from_x", 0.0)),
            _as_float(item.get("from_y", 0.0)),
            _as_float(item.get("damage", 0.0)),
            _as_float(item.get("alpha", 0.0)),
            bool(item.get("slow", False)),
        )

    def _rocket_sort_key(item: dict[str, Any]) -> tuple:
        return (
            str(item.get("kind", "")),
            _as_float(item.get("x", 0.0)),
            _as_float(item.get("y", 0.0)),
            _as_int(item.get("target_idx", -1)),
            _as_float(item.get("speed", 0.0)),
            _as_float(item.get("damage", 0.0)),
        )

    section_hashes = {
        "map": _hash_payload(snapshot.get("map")),
        "state": _hash_payload(snapshot.get("state")),
        "creeps": _hash_payload(snapshot.get("creeps")),
        "towers": _hash_payload(snapshot.get("towers")),
        "pulses": _hash_payload(snapshot.get("pulses")),
        "rockets": _hash_payload(snapshot.get("rockets")),
        "slot_trace": _hash_payload(snapshot.get("slot_trace")),
        "wave_ticks": _hash_payload(snapshot.get("wave_ticks")),
    }

    creeps_sorted = sorted(snapshot.get("creeps", []) or [], key=_creep_sort_key)
    towers_sorted = sorted(snapshot.get("towers", []) or [], key=_tower_sort_key)
    pulses_sorted = sorted(snapshot.get("pulses", []) or [], key=_pulse_sort_key)
    rockets_sorted = sorted(snapshot.get("rockets", []) or [], key=_rocket_sort_key)

    ordering_hashes = {
        "creeps_sorted": _hash_payload(creeps_sorted),
        "towers_sorted": _hash_payload(towers_sorted),
        "pulses_sorted": _hash_payload(pulses_sorted),
        "rockets_sorted": _hash_payload(rockets_sorted),
    }
    summary = {
        "bank": _as_int(getattr(state, "bank", 0)),
        "lives": _as_int(getattr(state, "lives", 0)),
        "score": _as_int(getattr(state, "score", 0)),
        "wave": _as_int(getattr(state, "level", 0)),
        "game_over": bool(getattr(state, "game_over", False)),
        "game_won": bool(getattr(state, "game_won", False)),
    }
    return {
        "hash": digest,
        "rng_state": _as_int(getattr(state, "rng_state", 0)),
        "rng_calls": _as_int(getattr(state, "rng_calls", 0)),
        "slot_trace": snapshot.get("slot_trace"),
        "wave_ticks": _as_int(wave_ticks),
        "section_hashes": section_hashes,
        "ordering_hashes": ordering_hashes,
        "summary": summary,
    }


def _resolve_map_path(map_path: str) -> Path:
    p = Path(map_path)
    if p.suffix:
        return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p
    if p.parent == Path("."):
        return Path(__file__).resolve().parents[3] / "data/maps" / f"{p.name}.json"
    return Path(__file__).resolve().parents[3] / p.with_suffix(".json")


def save_replay(path: str | Path, replay: Replay) -> None:
    payload = {
        "map_path": replay.map_path,
        "seed": int(replay.seed),
        "waves": [[action_to_dict(action) for action in wave] for wave in replay.waves],
        "state_hashes": replay.state_hashes,
        "final_summary": replay.final_summary,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_replay(path: str | Path) -> Replay:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "waves" not in data and "actions" in data:
        map_path = str(data.get("map_path", ""))
        if not map_path:
            raise ValueError("Replay missing map_path")
        resolved = _resolve_map_path(map_path)
        map_data = load_map_json(resolved)
        spec = action_space_spec(map_data)
        waves: list[list[Action]] = []
        current: list[Action] = []
        saw_start = False
        for raw_id in data.get("actions", []) or []:
            try:
                action = unflatten(int(raw_id), spec)
            except (TypeError, ValueError):
                action = Noop()
            current.append(action)
            if isinstance(action, StartWave):
                waves.append(current)
                current = []
                saw_start = True
        if not waves and not saw_start:
            raise ValueError("Replay actions contain no StartWave")
        seed = int(data.get("engine_seed", data.get("seed", 1)))
        return Replay(
            map_path=map_path,
            seed=seed,
            waves=waves,
            state_hashes=None,
            final_summary=None,
        )
    waves_raw = data.get("waves", []) or []
    waves: list[list[Action]] = []
    for wave in waves_raw:
        actions = [action_from_dict(action) for action in (wave or [])]
        waves.append(actions)
    return Replay(
        map_path=str(data.get("map_path", "")),
        seed=int(data.get("seed", 1)),
        waves=waves,
        state_hashes=data.get("state_hashes"),
        final_summary=data.get("final_summary"),
    )


def _apply_action(engine: Engine, spec, action: Action) -> None:
    if isinstance(action, Noop):
        return
    if isinstance(action, Place):
        if action.tower_type < 0 or action.tower_type >= len(spec.tower_kinds):
            return
        if action.cell < 0 or action.cell >= len(spec.cell_positions):
            return
        cell_x, cell_y = spec.cell_positions[action.cell]
        tower_kind = spec.tower_kinds[action.tower_type]
        engine.act("PLACE_TOWER", {"cell_x": cell_x, "cell_y": cell_y, "kind": tower_kind})
        return
    if isinstance(action, Upgrade):
        tower = _tower_from_slot(engine, spec, action.tower_id)
        if tower is None:
            return
        engine.act("UPGRADE_TOWER", {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)})
        return
    if isinstance(action, Sell):
        tower = _tower_from_slot(engine, spec, action.tower_id)
        if tower is None:
            return
        engine.act("SELL_TOWER", {"cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)})
        return
    if isinstance(action, SetMode):
        tower = _tower_from_slot(engine, spec, action.tower_id)
        if tower is None:
            return
        if action.mode < 0 or action.mode >= len(spec.target_modes):
            return
        mode = spec.target_modes[action.mode]
        engine.act(
            "SET_TARGET_MODE",
            {"mode": mode, "cell_x": tower.cell_x, "cell_y": tower.cell_y, "tower_id": id(tower)},
        )
        return
    if isinstance(action, StartWave):
        engine.act("NEXT_WAVE")
        return
    raise TypeError(f"Unknown action {action!r}")


def _tower_from_slot(engine: Engine, spec, tower_id: int):
    if tower_id < 0 or tower_id >= spec.max_towers:
        return None
    tower_slots = get_tower_slots(engine.state, spec.max_towers)
    return tower_slots[tower_id]


def _run_wave(engine: Engine, *, max_wave_ticks: int) -> int:
    ticks = 0
    while ticks < max_wave_ticks:
        if getattr(engine.state, "game_over", False):
            break
        if not getattr(engine.state, "creeps", []):
            break
        engine.step(engine.FRAME_DT)
        ticks += 1
    if ticks >= max_wave_ticks:
        logger.error("Wave simulation exceeded max_wave_ticks=%s", max_wave_ticks)
    return ticks


def run_replay_headless(
    replay: Replay,
    *,
    assert_deterministic: bool = True,
    max_wave_ticks: int = 20000,
) -> dict[str, Any]:
    resolved = _resolve_map_path(replay.map_path)
    map_data = load_map_json(resolved)
    engine = Engine(map_data)
    engine.reset()
    seed_state(engine.state, replay.seed)

    spec = action_space_spec(map_data)
    for wave_idx, wave_actions in enumerate(replay.waves):
        saw_start = False
        for action in wave_actions:
            if isinstance(action, StartWave):
                _apply_action(engine, spec, action)
                saw_start = True
                _run_wave(engine, max_wave_ticks=max_wave_ticks)
                break
            _apply_action(engine, spec, action)
        if not saw_start:
            raise ValueError(f"Wave {wave_idx} missing StartWave action")
        if getattr(engine.state, "game_over", False):
            break

    summary = {
        "bank": int(getattr(engine.state, "bank", 0)),
        "lives": int(getattr(engine.state, "lives", 0)),
        "score": int(getattr(engine.state, "score", 0)),
        "wave": int(getattr(engine.state, "level", 0)),
        "game_over": bool(getattr(engine.state, "game_over", False)),
        "game_won": bool(getattr(engine.state, "game_won", False)),
    }
    if assert_deterministic and replay.final_summary is not None:
        if replay.final_summary != summary:
            raise AssertionError(f"Replay summary mismatch: {summary} != {replay.final_summary}")
    return summary


def run_replay_gui(replay: Replay) -> None:
    from vectortd.gui.pyglet_app import run

    run(replay.map_path, replay=replay)
