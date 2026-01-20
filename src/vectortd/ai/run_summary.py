from __future__ import annotations

import csv
import json
import re
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


_FPS_RE = re.compile(r"\|\s+fps\s+\|\s+([0-9eE\.\+\-]+)")
_ITER_RE = re.compile(r"\|\s+iterations\s+\|\s+([0-9]+)")
_TIME_ELAPSED_RE = re.compile(r"\|\s+time_elapsed\s+\|\s+([0-9eE\.\+\-]+)")
_TOTAL_STEPS_RE = re.compile(r"\|\s+total_timesteps\s+\|\s+([0-9]+)")
_STEPS_PER_SEC_RE = re.compile(r"steps_per_sec=([0-9eE\.\+\-]+)")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = stats.mean(values)
    median = stats.median(values)
    stdev = stats.pstdev(values) if len(values) > 1 else 0.0
    if stdev == 0.0:
        skew = 0.0
        kurt = -3.0
    else:
        n = len(values)
        mu = mean
        m3 = sum((x - mu) ** 3 for x in values) / n
        m4 = sum((x - mu) ** 4 for x in values) / n
        skew = m3 / (stdev ** 3)
        kurt = m4 / (stdev ** 4) - 3.0
    cv = stdev / mean if mean != 0.0 else None
    return {
        "count": len(values),
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "skew": skew,
        "excess_kurtosis": kurt,
        "cv": cv,
    }


def _segment_means(values: list[float]) -> dict[str, float | str] | None:
    if not values:
        return None
    n = len(values)
    third = max(1, n // 3)
    start = values[:third]
    mid = values[third : 2 * third]
    end = values[2 * third :]
    start_mean = stats.mean(start) if start else None
    mid_mean = stats.mean(mid) if mid else None
    end_mean = stats.mean(end) if end else None
    means = {"start": start_mean, "mid": mid_mean, "end": end_mean}
    highest_segment = None
    highest_value = None
    for key, value in means.items():
        if value is None:
            continue
        if highest_value is None or value > highest_value:
            highest_value = value
            highest_segment = key
    return {
        "start_mean": start_mean,
        "mid_mean": mid_mean,
        "end_mean": end_mean,
        "highest_segment": highest_segment or "unknown",
    }


def _parse_console_log(path: Path) -> dict[str, Any]:
    fps_vals: list[float] = []
    steps_per_sec_vals: list[float] = []
    iter_vals: list[int] = []
    time_elapsed_vals: list[float] = []
    total_steps_vals: list[int] = []
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = _FPS_RE.search(line)
            if match:
                value = _safe_float(match.group(1))
                if value is not None:
                    fps_vals.append(value)
            match = _STEPS_PER_SEC_RE.search(line)
            if match:
                value = _safe_float(match.group(1))
                if value is not None:
                    steps_per_sec_vals.append(value)
            match = _ITER_RE.search(line)
            if match:
                iter_vals.append(int(match.group(1)))
            match = _TIME_ELAPSED_RE.search(line)
            if match:
                value = _safe_float(match.group(1))
                if value is not None:
                    time_elapsed_vals.append(value)
            match = _TOTAL_STEPS_RE.search(line)
            if match:
                total_steps_vals.append(int(match.group(1)))
    overall_steps_per_sec = None
    if time_elapsed_vals and total_steps_vals:
        elapsed = time_elapsed_vals[-1]
        steps = total_steps_vals[-1]
        if elapsed > 0:
            overall_steps_per_sec = steps / elapsed
    return {
        "fps_stats": _compute_stats(fps_vals),
        "fps_segment_means": _segment_means(fps_vals),
        "steps_per_sec_stats": _compute_stats(steps_per_sec_vals),
        "overall_steps_per_sec": overall_steps_per_sec,
        "iterations_count": len(iter_vals),
        "timesteps_last": total_steps_vals[-1] if total_steps_vals else None,
    }


def _parse_monitor_dir(path: Path) -> dict[str, Any]:
    rewards: list[float] = []
    lengths: list[float] = []
    if not path.exists():
        return {}
    for file_path in sorted(path.glob("*.csv.monitor.csv")):
        with file_path.open(encoding="utf-8") as handle:
            header = handle.readline().strip()
            if header.startswith("#"):
                cols = handle.readline().strip()
            else:
                cols = header
            if not cols:
                continue
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                reward = _safe_float(parts[0])
                length = _safe_float(parts[1])
                if reward is not None:
                    rewards.append(reward)
                if length is not None:
                    lengths.append(length)
    return {
        "episode_reward_stats": _compute_stats(rewards),
        "episode_length_stats": _compute_stats(lengths),
        "best_episode_reward": max(rewards) if rewards else None,
        "episode_count": len(rewards),
    }


def _parse_eval_npz(path: Path) -> dict[str, Any]:
    if not path.exists() or np is None:
        return {}
    data = np.load(path)
    results = data.get("results")
    ep_lengths = data.get("ep_lengths")
    timesteps = data.get("timesteps")
    output: dict[str, Any] = {}
    if results is not None:
        flat = [float(x) for x in results.reshape(-1).tolist()]
        output["episode_reward_stats"] = _compute_stats(flat)
        output["best_episode_reward"] = max(flat) if flat else None
    if ep_lengths is not None:
        flat_lengths = [float(x) for x in ep_lengths.reshape(-1).tolist()]
        output["episode_length_stats"] = _compute_stats(flat_lengths)
    if timesteps is not None:
        output["timesteps"] = [int(x) for x in timesteps.tolist()]
    return output


def _parse_csv_stats(path: Path, fields: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: dict[str, list[float]] = {field: [] for field in fields}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in fields:
                value = _safe_float(row.get(field, ""))
                if value is not None:
                    values[field].append(value)
    output: dict[str, Any] = {}
    for field, field_values in values.items():
        output[field] = {
            "stats": _compute_stats(field_values),
            "best": max(field_values) if field_values else None,
        }
    output["row_count"] = len(next(iter(values.values()), []))
    return output


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_run_summary(
    run_dir: Path,
    *,
    algorithm: str,
    console_log: Path | None = None,
    monitor_train_dir: Path | None = None,
    monitor_eval_dir: Path | None = None,
    eval_npz_path: Path | None = None,
    train_log_csv: Path | None = None,
    eval_log_csv: Path | None = None,
    error: dict[str, str] | None = None,
) -> Path:
    summary_path = run_dir / "run_summary.json"
    payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}

    now = _iso_utc_now()
    payload.setdefault("schema_version", 1)
    payload.setdefault("created_at_utc", now)
    payload["updated_at_utc"] = now
    payload["run_dir"] = str(run_dir)
    payload["algorithm"] = algorithm

    metadata = _load_run_metadata(run_dir)
    if metadata is not None:
        training = metadata.get("training", {})
        payload["metadata"] = {
            "total_timesteps": training.get("total_timesteps"),
            "actual_timesteps": training.get("actual_timesteps"),
            "step_unit": training.get("step_unit"),
        }

    sources: dict[str, Any] = {}
    if console_log:
        sources["console_log"] = str(console_log)
    if monitor_train_dir:
        sources["monitor_train_dir"] = str(monitor_train_dir)
    if monitor_eval_dir:
        sources["monitor_eval_dir"] = str(monitor_eval_dir)
    if eval_npz_path:
        sources["eval_npz_path"] = str(eval_npz_path)
    if train_log_csv:
        sources["train_log_csv"] = str(train_log_csv)
    if eval_log_csv:
        sources["eval_log_csv"] = str(eval_log_csv)
    payload["sources"] = sources

    if console_log:
        payload["throughput"] = _parse_console_log(console_log)

    learning: dict[str, Any] = {}
    if monitor_train_dir:
        learning["train"] = _parse_monitor_dir(monitor_train_dir)
    if monitor_eval_dir:
        learning["eval"] = _parse_monitor_dir(monitor_eval_dir)
    if eval_npz_path:
        learning["eval_npz"] = _parse_eval_npz(eval_npz_path)
    if train_log_csv:
        learning["train_updates"] = _parse_csv_stats(train_log_csv, ["mean_reward", "mean_return"])
    if eval_log_csv:
        learning["eval_updates"] = _parse_csv_stats(eval_log_csv, ["mean_return", "mean_score", "max_score"])
    if learning:
        payload["learning"] = learning

    if error:
        payload["error"] = {"type": error.get("type"), "message": error.get("message")}

    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path
