from __future__ import annotations

import argparse
import atexit
import logging
from pathlib import Path
import queue
import sys
import threading
from vectortd.ai.actions import Noop, StartWave, get_tower_slots
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.policies.baseline import make_policy
from vectortd.io.replay import Replay, build_state_check, save_replay


logger = logging.getLogger(__name__)


class _AsyncFileWriter:
    def __init__(self, path: Path) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._handle = path.open("w", encoding="utf-8")
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            self._handle.write(data)
            self._queue.task_done()
        self._handle.flush()
        self._handle.close()

    def write(self, data: str) -> None:
        self._queue.put(data)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()


class AsyncTee:
    def __init__(self, console_stream, writer: _AsyncFileWriter) -> None:
        self._console = console_stream
        self._writer = writer

    def write(self, data: str) -> int:
        self._console.write(data)
        self._console.flush()
        self._writer.write(data)
        return len(data)

    def flush(self) -> None:
        self._console.flush()


def _next_log_path(run_dir: Path) -> Path:
    max_idx = 0
    for path in run_dir.glob("log_*.txt"):
        stem = path.stem
        if not stem.startswith("log_"):
            continue
        suffix = stem[4:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return run_dir / f"log_{max_idx + 1}.txt"


def _configure_logging(*, verbose: bool, log_dir: Path | None) -> Path | None:
    handlers: list[logging.Handler] = []
    log_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _next_log_path(log_dir)
        writer = _AsyncFileWriter(log_path)
        sys.stdout = AsyncTee(sys.stdout, writer)
        sys.stderr = AsyncTee(sys.stderr, writer)
        handlers.append(logging.StreamHandler(sys.stdout))
        atexit.register(writer.close)
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
        handlers=handlers,
    )
    if log_path is not None:
        logger.info("log_path=%s", log_path)
    return log_path


def _next_replay_path(replay_dir: Path) -> Path:
    replay_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in replay_dir.glob("replay_*.json"):
        stem = path.stem
        if not stem.startswith("replay_"):
            continue
        suffix = stem[7:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return replay_dir / f"replay_{max_idx + 1}.json"


def _save_replay(
    env: VectorTDEventEnv,
    replay_dir: Path,
    *,
    state_checks: list[dict] | None = None,
) -> Path | None:
    if env.engine is None:
        return None
    if not env.episode_actions:
        logger.warning("No completed waves; replay not saved")
        return None
    if env.map_path is None:
        logger.warning("Missing map path; replay not saved")
        return None
    seed = env.episode_seed if env.episode_seed is not None else 1
    state = env.engine.state
    summary = {
        "bank": int(getattr(state, "bank", 0)),
        "lives": int(getattr(state, "lives", 0)),
        "score": int(getattr(state, "score", 0)),
        "wave": int(getattr(state, "level", 0)),
        "game_over": bool(getattr(state, "game_over", False)),
        "game_won": bool(getattr(state, "game_won", False)),
    }
    replay = Replay(
        map_path=env.map_path,
        seed=int(seed),
        waves=list(env.episode_actions),
        state_hashes=state_checks,
        final_summary=summary,
    )
    replay_path = _next_replay_path(replay_dir)
    save_replay(replay_path, replay)
    logger.info("replay_path=%s", replay_path)
    return replay_path


def _all_upgrades_done(env: VectorTDEventEnv) -> bool:
    if env.engine is None:
        return True
    for tower in getattr(env.engine.state, "towers", []) or []:
        if int(getattr(tower, "level", 0)) < 10:
            return False
    return True


def _tower_kind_label(kind: str) -> str:
    labels = {
        "green": "gl1",
        "green2": "gl2",
        "green3": "gl3",
        "red_refractor": "red refractor",
        "red_spammer": "red spam.",
        "red_rockets": "red rockets",
        "purple1": "purple pulse 1",
        "purple2": "purple pulse 2",
        "purple3": "purple pulse 3",
        "blue1": "blue 1",
        "blue2": "blue 2",
    }
    return labels.get(kind, kind)


def _tower_summary(env: VectorTDEventEnv) -> str:
    if env.engine is None:
        return "none"
    towers = get_tower_slots(env.engine.state, env.max_towers)
    entries: list[str] = []
    for tower in towers:
        if tower is None:
            continue
        kind = str(getattr(tower, "kind", ""))
        level = int(getattr(tower, "level", 0))
        entries.append(f"{_tower_kind_label(kind)} ({level})")
    return ", ".join(entries) if entries else "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "smoke"], default="baseline")
    ap.add_argument("--policy", choices=["heuristic", "random"], default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-waves", type=int, default=None)
    ap.add_argument("--max-build-actions", type=int, default=None)
    ap.add_argument("--save-replay", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--stop-when-upgraded", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[3]
    if args.mode == "smoke":
        if args.policy is None:
            args.policy = "heuristic"
        if args.verbose is None:
            args.verbose = True
        if args.save_replay is None:
            args.save_replay = True
        if args.stop_when_upgraded is None:
            args.stop_when_upgraded = True
        if args.log_dir is None:
            args.log_dir = str(root_dir / "runs" / "smoke")
        if args.replay_dir is None:
            args.replay_dir = str(root_dir / "runs" / "smoke" / "replays")
    else:
        if args.policy is None:
            args.policy = "heuristic"

    if args.map is None:
        args.map = "switchback"
    if args.seed is None:
        args.seed = 123
    if args.max_steps is None:
        args.max_steps = 500
    if args.max_build_actions is None:
        args.max_build_actions = 100
    if args.verbose is None:
        args.verbose = False
    if args.save_replay is None:
        args.save_replay = False
    if args.stop_when_upgraded is None:
        args.stop_when_upgraded = False
    if args.save_replay and args.replay_dir is None:
        args.replay_dir = str(root_dir / "runs" / "baseline" / "replays")

    log_dir = Path(args.log_dir) if args.log_dir else None
    replay_dir = Path(args.replay_dir) if args.replay_dir else None
    _configure_logging(verbose=args.verbose, log_dir=log_dir)

    env = VectorTDEventEnv(max_build_actions=args.max_build_actions)
    env.reset(seed=args.seed, options={"map_path": args.map})

    policy = make_policy(args.policy, seed=args.seed, verbose=args.verbose)
    policy.reset(env)

    state_checks: list[dict] = []
    steps = 0
    waves = 0
    done = False
    obs_dict: dict | None = None

    while not done and steps < args.max_steps:
        if args.max_waves is not None and waves >= args.max_waves:
            break
        action = policy.next_action(env)
        if action is None:
            action = Noop()

        is_start_wave = isinstance(action, StartWave)
        if not is_start_wave and isinstance(action, int) and env.action_spec is not None:
            is_start_wave = action == env.action_spec.offsets.start_wave

        pre_check = None
        if is_start_wave and args.save_replay and env.engine is not None and env.action_spec is not None:
            if args.verbose:
                logger.info("towers before wave: %s", _tower_summary(env))
            pre_check = build_state_check(
                env.engine.state,
                env.map_data,
                env.action_spec,
                wave_ticks=0,
            )

        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs_dict = env.last_obs
        steps += 1

        if is_start_wave:
            waves += 1
            if args.verbose:
                logger.info(
                    "wave done: ticks=%s timeout=%s lives=%s score=%s bank=%s wave=%s",
                    info.get("wave_ticks"),
                    info.get("timeout"),
                    (obs_dict or {}).get("lives"),
                    (obs_dict or {}).get("score"),
                    (obs_dict or {}).get("bank"),
                    (obs_dict or {}).get("wave"),
                )
            if pre_check is not None and env.engine is not None and env.action_spec is not None:
                wave_idx = max(0, len(env.episode_actions) - 1)
                wave_ticks = int(info.get("wave_ticks", 0) or 0)
                post_check = build_state_check(
                    env.engine.state,
                    env.map_data,
                    env.action_spec,
                    wave_ticks=wave_ticks,
                )
                state_checks.append(
                    {
                        "wave_index": wave_idx,
                        "pre": pre_check,
                        "post": post_check,
                    }
                )

        if info.get("invalid_action") and args.verbose:
            logger.info("invalid action %s", action)
        if args.stop_when_upgraded and _all_upgrades_done(env):
            logger.info("all upgrades complete after %s steps", steps)
            break

    if obs_dict is None:
        obs_dict = {}
    summary = (
        f"summary: steps={steps} waves={waves} "
        f"wave={obs_dict.get('wave')} lives={obs_dict.get('lives')} "
        f"score={obs_dict.get('score')} bank={obs_dict.get('bank')} done={done}"
    )
    if args.verbose:
        logger.info(summary)
    else:
        print(summary)

    if args.save_replay and replay_dir is not None:
        _save_replay(env, replay_dir, state_checks=state_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
