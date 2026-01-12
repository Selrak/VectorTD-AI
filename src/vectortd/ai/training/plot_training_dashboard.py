from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _find_latest_log(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_latest_run_dir(root: Path) -> Path | None:
    runs_root = root / "runs" / "ppo"
    candidates = [path for path in runs_root.glob("run_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _select_run(rows: list[dict[str, str]], run_id: str | None) -> tuple[list[dict[str, str]], str | None]:
    if not rows:
        return rows, run_id
    if run_id:
        filtered = [row for row in rows if row.get("run_id") == run_id]
        return filtered, run_id
    if "run_id" in rows[-1]:
        last_run = rows[-1].get("run_id") or None
        if last_run:
            filtered = [row for row in rows if row.get("run_id") == last_run]
            return filtered, last_run
    return rows, run_id


def _column(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        try:
            values.append(float(raw))
        except ValueError:
            values.append(0.0)
    return values


def _plot_series(ax, x: Iterable[float], y: Iterable[float], label: str) -> None:
    ax.plot(list(x), list(y), label=label)
    ax.legend(loc="best")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--eval-log", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit("matplotlib is required to build the training dashboard.") from exc

    root_dir = _resolve_root()
    run_dir = _find_latest_run_dir(root_dir)
    train_log = Path(args.train_log) if args.train_log else None
    eval_log = Path(args.eval_log) if args.eval_log else None
    if train_log is None and run_dir is not None:
        candidate = run_dir / "train_log.csv"
        if candidate.exists():
            train_log = candidate
    if eval_log is None and run_dir is not None:
        candidate = run_dir / "eval_log.csv"
        if candidate.exists():
            eval_log = candidate
    if train_log is None:
        train_log = _find_latest_log(root_dir / "runs" / "ppo", f"train_log_{args.map}_*.csv")
    if eval_log is None:
        eval_log = _find_latest_log(root_dir / "runs" / "ppo", f"eval_log_{args.map}_*.csv")
    if train_log is None and eval_log is None:
        raise SystemExit("No training or eval logs found.")

    out_dir = Path(args.out_dir) if args.out_dir else root_dir / "runs" / "ppo" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    run_id = args.run_id
    if train_log is not None and train_log.exists():
        train_rows = _read_csv(train_log)
        train_rows, run_id = _select_run(train_rows, run_id)
    if eval_log is not None and eval_log.exists():
        eval_rows = _read_csv(eval_log)
        eval_rows, run_id = _select_run(eval_rows, run_id)

    image_paths = []

    if train_rows:
        steps = _column(train_rows, "step")
        fig, ax = plt.subplots()
        _plot_series(ax, steps, _column(train_rows, "policy_loss"), "policy_loss")
        _plot_series(ax, steps, _column(train_rows, "value_loss"), "value_loss")
        ax.set_title("Training Losses")
        ax.set_xlabel("steps")
        ax.set_ylabel("loss")
        loss_path = out_dir / "train_losses.png"
        fig.tight_layout()
        fig.savefig(loss_path, dpi=140)
        image_paths.append(loss_path.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        _plot_series(ax, steps, _column(train_rows, "entropy"), "entropy")
        ax.set_title("Policy Entropy")
        ax.set_xlabel("steps")
        ax.set_ylabel("entropy")
        entropy_path = out_dir / "train_entropy.png"
        fig.tight_layout()
        fig.savefig(entropy_path, dpi=140)
        image_paths.append(entropy_path.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        _plot_series(ax, steps, _column(train_rows, "mean_reward"), "mean_reward")
        _plot_series(ax, steps, _column(train_rows, "mean_return"), "mean_return")
        _plot_series(ax, steps, _column(train_rows, "mean_value"), "mean_value")
        ax.set_title("Reward/Return/Value")
        ax.set_xlabel("steps")
        ax.set_ylabel("value")
        reward_path = out_dir / "train_returns.png"
        fig.tight_layout()
        fig.savefig(reward_path, dpi=140)
        image_paths.append(reward_path.name)
        plt.close(fig)

    if eval_rows:
        eval_steps = _column(eval_rows, "step")
        fig, ax = plt.subplots()
        _plot_series(ax, eval_steps, _column(eval_rows, "mean_score"), "mean_score")
        _plot_series(ax, eval_steps, _column(eval_rows, "max_score"), "max_score")
        ax.set_title("Eval Score")
        ax.set_xlabel("steps")
        ax.set_ylabel("score")
        score_path = out_dir / "eval_score.png"
        fig.tight_layout()
        fig.savefig(score_path, dpi=140)
        image_paths.append(score_path.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        _plot_series(ax, eval_steps, _column(eval_rows, "mean_wave"), "mean_wave")
        _plot_series(ax, eval_steps, _column(eval_rows, "max_wave"), "max_wave")
        ax.set_title("Eval Wave")
        ax.set_xlabel("steps")
        ax.set_ylabel("wave")
        wave_path = out_dir / "eval_wave.png"
        fig.tight_layout()
        fig.savefig(wave_path, dpi=140)
        image_paths.append(wave_path.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        _plot_series(ax, eval_steps, _column(eval_rows, "win_rate"), "win_rate")
        _plot_series(ax, eval_steps, _column(eval_rows, "loss_rate"), "loss_rate")
        ax.set_title("Eval Win/Loss")
        ax.set_xlabel("steps")
        ax.set_ylabel("rate")
        win_path = out_dir / "eval_win_loss.png"
        fig.tight_layout()
        fig.savefig(win_path, dpi=140)
        image_paths.append(win_path.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        _plot_series(ax, eval_steps, _column(eval_rows, "mean_lives"), "mean_lives")
        _plot_series(ax, eval_steps, _column(eval_rows, "mean_actions_per_wave"), "mean_actions_per_wave")
        ax.set_title("Eval Lives/Actions")
        ax.set_xlabel("steps")
        ax.set_ylabel("value")
        actions_path = out_dir / "eval_lives_actions.png"
        fig.tight_layout()
        fig.savefig(actions_path, dpi=140)
        image_paths.append(actions_path.name)
        plt.close(fig)

    title = f"VectorTD Training Dashboard ({args.map})"
    if run_id:
        title = f"{title} - run {run_id}"
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]
    if train_log is not None:
        html_lines.append(f"<p>train_log: {train_log}</p>")
    if eval_log is not None:
        html_lines.append(f"<p>eval_log: {eval_log}</p>")
    for image in image_paths:
        html_lines.append(f"<div><img src=\"{image}\" alt=\"{image}\"></div>")
    html_lines.extend(["</body>", "</html>"])
    html_path = out_dir / "index.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"dashboard={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
