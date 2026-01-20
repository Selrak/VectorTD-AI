from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _find_git_root(start_dir: Path) -> Path | None:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(root_dir: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root_dir), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata(start_dir: Path) -> dict[str, Any] | None:
    root_dir = _find_git_root(start_dir)
    if root_dir is None:
        return None
    info: dict[str, Any] = {"root": str(root_dir)}
    commit = _run_git(root_dir, ["rev-parse", "HEAD"])
    if commit:
        info["commit"] = commit
    branch = _run_git(root_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        info["branch"] = branch
    status = _run_git(root_dir, ["status", "--porcelain"])
    if status is not None:
        lines = [line for line in status.splitlines() if line.strip()]
        info["is_dirty"] = bool(lines)
        info["dirty_paths"] = len(lines)
    return info


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def write_run_metadata(
    run_dir: Path,
    *,
    total_timesteps: int,
    step_unit: str = "timesteps",
    training: dict[str, Any] | None = None,
    algorithm: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    path = run_dir / "run_metadata.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}

    now = _iso_utc_now()
    payload.setdefault("schema_version", 1)
    payload.setdefault("created_at_utc", now)
    payload["updated_at_utc"] = now
    payload["run_id"] = run_dir.name
    payload["run_dir"] = str(run_dir)
    payload["cwd"] = str(Path.cwd())
    payload["argv"] = list(sys.argv)
    payload["python"] = {
        "version": platform.python_version(),
        "executable": sys.executable,
    }
    payload["host"] = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    payload["process"] = {
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, "getppid") else None,
    }

    git_info = _git_metadata(run_dir)
    if git_info:
        payload["git"] = git_info

    training_payload = _ensure_dict(payload.get("training"))
    training_payload["total_timesteps"] = int(total_timesteps)
    training_payload["step_unit"] = step_unit
    if training:
        training_payload.update(training)
    payload["training"] = training_payload

    if algorithm:
        algorithm_payload = _ensure_dict(payload.get("algorithm"))
        algorithm_payload.update(algorithm)
        payload["algorithm"] = algorithm_payload

    if extras:
        extras_payload = _ensure_dict(payload.get("extras"))
        extras_payload.update(extras)
        payload["extras"] = extras_payload

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
