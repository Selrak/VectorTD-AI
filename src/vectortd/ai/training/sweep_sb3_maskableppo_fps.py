from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(",")]
    values = []
    for part in parts:
        if part:
            values.append(int(part))
    return values


def _read_cpu_max_freq_khz(cpu_id: int) -> int | None:
    cpu_dir = Path("/sys/devices/system/cpu") / f"cpu{cpu_id}" / "cpufreq"
    for name in ("cpuinfo_max_freq", "scaling_max_freq"):
        path = cpu_dir / name
        if not path.exists():
            continue
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
    return None


def _detect_p_core_cpus() -> list[int]:
    cpu_root = Path("/sys/devices/system/cpu")
    cpu_ids: list[int] = []
    for entry in cpu_root.iterdir():
        name = entry.name
        if name.startswith("cpu") and name[3:].isdigit():
            cpu_ids.append(int(name[3:]))
    cpu_ids.sort()
    freq_by_cpu: dict[int, int] = {}
    for cpu_id in cpu_ids:
        freq = _read_cpu_max_freq_khz(cpu_id)
        if freq is not None:
            freq_by_cpu[cpu_id] = freq
    if not freq_by_cpu:
        return []
    max_freq = max(freq_by_cpu.values())
    if max_freq <= 0:
        return []
    threshold = int(max_freq * 0.95)
    return sorted([cpu_id for cpu_id, freq in freq_by_cpu.items() if freq >= threshold])


def _warmup_worker(warmup_sec: float, cpu_id: int | None) -> None:
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu_id})
        except OSError:
            pass
    end_time = time.perf_counter() + warmup_sec
    value = 0.0
    while time.perf_counter() < end_time:
        value = (value + 1.0) * 1.0000001
    _ = value


def _warmup_cpus(warmup_sec: float, cpu_ids: list[int], workers: int) -> None:
    if warmup_sec <= 0 or workers <= 0:
        return
    import multiprocessing as mp

    if cpu_ids:
        selected = [cpu_ids[idx % len(cpu_ids)] for idx in range(workers)]
    else:
        selected = [None] * workers
    procs = []
    for cpu_id in selected:
        proc = mp.Process(target=_warmup_worker, args=(warmup_sec, cpu_id))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def _extract_fps_values(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    fps_values: list[float] = []
    pattern = re.compile(r"\|\s+fps\s+\|\s+([0-9]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            fps_values.append(float(match.group(1)))
    return fps_values


def _score_fps(
    fps_values: list[float], *, skip: int, window: int, mode: str
) -> float | None:
    if not fps_values:
        return None
    values = fps_values[skip:] if skip > 0 else fps_values
    if not values:
        return None
    if window > 0:
        values = values[-window:]
    if not values:
        return None
    if mode == "median":
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2 == 1:
            return values_sorted[mid]
        return 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
    if mode == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unknown score_mode={mode}")


def _next_sweep_dir(root_dir: Path) -> Path:
    runs_root = root_dir / "runs" / "sb3_maskableppo" / "sweep"
    runs_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("run_"):
            continue
        suffix = name[4:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    run_dir = runs_root / f"run_{max_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_cmd(
    *,
    map_name: str,
    seed: int,
    num_envs: int,
    n_steps: int,
    batch_size: int,
    total_timesteps: int,
    eval_envs: int,
    eval_freq: int,
    checkpoint_freq: int,
    run_dir: Path,
    debug: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vectortd.ai.training.train_sb3_maskableppo",
        "--map",
        map_name,
        "--seed",
        str(seed),
        "--num-envs",
        str(num_envs),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--total-timesteps",
        str(total_timesteps),
        "--eval-envs",
        str(eval_envs),
        "--eval-freq",
        str(eval_freq),
        "--checkpoint-freq",
        str(checkpoint_freq),
        "--run-dir",
        str(run_dir),
    ]
    if debug:
        cmd.append("--debug")
    return cmd


def _run_command(cmd: list[str]) -> int:
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _ensure_iterable(values: Iterable[int]) -> list[int]:
    return sorted(set(int(value) for value in values if int(value) > 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num-envs-list", default="12,14,16")
    ap.add_argument("--n-steps-list", default="64,128,256")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--total-timesteps", type=int, default=50_000)
    ap.add_argument("--eval-envs", type=int, default=1)
    ap.add_argument("--eval-freq", type=int, default=100_000_000)
    ap.add_argument("--checkpoint-freq", type=int, default=100_000_000)
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--score-skip", type=int, default=2)
    ap.add_argument("--score-window", type=int, default=5)
    ap.add_argument("--score-mode", choices=["median", "mean"], default="median")
    ap.add_argument("--require-divisible", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmup-sec", type=float, default=6.0)
    ap.add_argument("--warmup-cores", choices=["pcores", "all", "none"], default="pcores")
    ap.add_argument("--warmup-workers", type=int, default=0)
    args = ap.parse_args()

    num_envs_list = _ensure_iterable(_parse_int_list(args.num_envs_list))
    n_steps_list = _ensure_iterable(_parse_int_list(args.n_steps_list))
    if not num_envs_list or not n_steps_list:
        raise SystemExit("num-envs-list and n-steps-list must contain at least one value")

    root_dir = _resolve_root()
    sweep_dir = _next_sweep_dir(root_dir)

    cpu_ids: list[int] = []
    if args.warmup_cores == "pcores":
        cpu_ids = _detect_p_core_cpus()
    elif args.warmup_cores == "all":
        cpu_ids = list(range(os.cpu_count() or 1))
    if args.warmup_cores != "none":
        warmup_workers = args.warmup_workers or (len(cpu_ids) if cpu_ids else (os.cpu_count() or 1))
        _warmup_cpus(args.warmup_sec, cpu_ids, warmup_workers)

    results: list[dict[str, object]] = []
    for num_envs in num_envs_list:
        for n_steps in n_steps_list:
            total_timesteps = int(args.total_timesteps)
            batch_size = int(args.batch_size)
            rollout = num_envs * n_steps
            required_iters = max(1, args.score_skip + args.score_window)
            min_timesteps = required_iters * rollout
            if total_timesteps < min_timesteps:
                total_timesteps = min_timesteps
            if rollout < batch_size:
                results.append(
                    {
                        "num_envs": num_envs,
                        "n_steps": n_steps,
                        "status": "skipped",
                        "reason": "rollout_smaller_than_batch_size",
                    }
                )
                continue
            if args.require_divisible and (rollout % batch_size) != 0:
                results.append(
                    {
                        "num_envs": num_envs,
                        "n_steps": n_steps,
                        "status": "skipped",
                        "reason": "rollout_not_divisible_by_batch_size",
                    }
                )
                continue

            config_dir = sweep_dir / f"envs{num_envs}_steps{n_steps}_bs{batch_size}"
            config_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_cmd(
                map_name=args.map,
                seed=args.seed,
                num_envs=num_envs,
                n_steps=n_steps,
                batch_size=batch_size,
                total_timesteps=total_timesteps,
                eval_envs=args.eval_envs,
                eval_freq=args.eval_freq,
                checkpoint_freq=args.checkpoint_freq,
                run_dir=config_dir,
                debug=args.debug,
            )
            returncode = _run_command(cmd)
            console_log = config_dir / "console.log"
            fps_values = _extract_fps_values(console_log)
            score = _score_fps(
                fps_values, skip=args.score_skip, window=args.score_window, mode=args.score_mode
            )
            results.append(
                {
                    "num_envs": num_envs,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "total_timesteps": total_timesteps,
                    "returncode": returncode,
                    "fps_values": fps_values,
                    "score": score,
                    "console_log": str(console_log),
                    "status": "ok" if returncode == 0 else "failed",
                }
            )

    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    best = None
    for entry in results:
        if entry.get("status") != "ok":
            continue
        score = entry.get("score")
        if score is None:
            continue
        if best is None or float(score) > float(best.get("score") or 0.0):
            best = entry

    print(f"sweep_dir={sweep_dir}")
    if best is None:
        print("best_config=none")
        return 1
    print(
        "best_config=num_envs={num_envs} n_steps={n_steps} batch_size={batch_size} fps_score={score}".format(
            num_envs=best.get("num_envs"),
            n_steps=best.get("n_steps"),
            batch_size=best.get("batch_size"),
            score=best.get("score"),
        )
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
