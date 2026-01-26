from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_SUPPORTED_SCHEMA_VERSIONS = {1}
_ALLOWED_KEYS: dict[str, Any] = {
    "schema_version": None,
    "run": {
        "total_timesteps": None,
        "num_envs": None,
        "num_eval_envs": None,
        "log_interval_sec": None,
    },
    "env": {
        "action_space_kind": None,
        "max_build_actions": None,
        "max_wave_ticks": None,
        "deterministic_eval": None,
    },
    "masking": {
        "place_cell_top_k": None,
    },
    "reward": {
        "score_weight": None,
        "score_delta_clip": None,
        "life_loss_penalty": None,
        "no_life_loss_bonus": None,
        "terminal_loss_penalty": None,
        "terminal_win_bonus": None,
        "build_step_penalty": None,
        "noop_penalty": None,
        "set_mode_penalty": None,
        "set_mode_noop_penalty": None,
    },
    "sb3": {
        "algo": None,
        "policy": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "gamma": None,
        "gae_lambda": None,
        "clip_range": None,
        "ent_coef": None,
        "vf_coef": None,
        "max_grad_norm": None,
        "policy_kwargs": {
            "net_arch": None,
            "activation_fn": None,
            "ortho_init": None,
        },
    },
    "callbacks": {
        "eval_freq_steps": None,
        "n_eval_episodes": None,
        "save_freq_steps": None,
    },
}


def load_json_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a JSON object: {p}")
    _validate_config(payload)
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(cfg: dict[str, Any], overrides_list: list[str] | None) -> dict[str, Any]:
    if not overrides_list:
        return cfg

    out = cfg
    for item in overrides_list:
        if "=" not in item:
            raise ValueError(f"override must contain '=': {item}")
        path_str, value_str = item.split("=", 1)
        if not path_str:
            raise ValueError(f"override path empty: {item}")
        keys = path_str.split(".")
        if any(not key for key in keys):
            raise ValueError(f"override path has empty segment: {item}")
        value = _cast_scalar(value_str)

        cursor = out
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return out


def dump_effective_config(run_dir: str | Path, cfg: dict[str, Any]) -> str:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()

    eff_path = run_path / "effective_config.json"
    with eff_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)

    (run_path / "effective_config.sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def _cast_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in parent:
        raise ValueError(f"missing '{key}' section in config")
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"config '{key}' must be a JSON object")
    return value


def _require_number(parent: dict[str, Any], key: str) -> float:
    if key not in parent:
        raise ValueError(f"missing '{key}' in config")
    value = parent[key]
    if not _is_number(value):
        raise ValueError(f"config '{key}' must be a number")
    return float(value)


def _validate_config(cfg: dict[str, Any]) -> None:
    unknown = _find_unknown_keys(cfg, _ALLOWED_KEYS, path="")
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config keys: {unknown_str}")

    schema_version = cfg.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    run_cfg = _require_dict(cfg, "run")
    num_envs = _require_number(run_cfg, "num_envs")
    num_eval_envs = _require_number(run_cfg, "num_eval_envs")
    total_timesteps = _require_number(run_cfg, "total_timesteps")
    if num_envs < 1:
        raise ValueError("run.num_envs must be >= 1")
    if num_eval_envs < 1:
        raise ValueError("run.num_eval_envs must be >= 1")
    if total_timesteps <= 0:
        raise ValueError("run.total_timesteps must be > 0")

    sb3_cfg = _require_dict(cfg, "sb3")
    batch_size = _require_number(sb3_cfg, "batch_size")
    n_steps = _require_number(sb3_cfg, "n_steps")
    if batch_size > n_steps * num_envs:
        raise ValueError("sb3.batch_size must be <= sb3.n_steps * run.num_envs")

    masking_cfg = _require_dict(cfg, "masking")
    place_cell_top_k = masking_cfg.get("place_cell_top_k")
    if place_cell_top_k is not None:
        if not _is_number(place_cell_top_k):
            raise ValueError("masking.place_cell_top_k must be a number or null")
        if place_cell_top_k < 0:
            raise ValueError("masking.place_cell_top_k must be >= 0 or null")

    env_cfg = _require_dict(cfg, "env")
    action_space_kind = env_cfg.get("action_space_kind")
    if action_space_kind is not None:
        if not isinstance(action_space_kind, str):
            raise ValueError("env.action_space_kind must be a string or null")
        if action_space_kind not in ("legacy", "discrete_k"):
            raise ValueError("env.action_space_kind must be 'legacy' or 'discrete_k'")


def _find_unknown_keys(value: Any, allowed: Any, *, path: str) -> list[str]:
    if not isinstance(value, dict) or not isinstance(allowed, dict):
        return []
    unknown: list[str] = []
    for key, sub_value in value.items():
        if key not in allowed:
            unknown.append(f"{path}{key}" if path else key)
            continue
        sub_allowed = allowed[key]
        if isinstance(sub_value, dict) and isinstance(sub_allowed, dict):
            child_path = f"{path}{key}."
            unknown.extend(_find_unknown_keys(sub_value, sub_allowed, path=child_path))
    return unknown
