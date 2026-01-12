from __future__ import annotations

from pathlib import Path
import argparse
import json
import platform
import subprocess

from vectortd.gui.pyglet_app import run, run_replay
from vectortd.testing.definitions import load_test_definitions, resolve_map_path, resolve_tests_dir


def _replay_state_path(root: Path) -> Path:
    return root / "runs" / "smoke" / "last_replay.txt"


def _load_last_replay(root: Path) -> Path | None:
    path = _replay_state_path(root)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    last = Path(text)
    if last.exists():
        return last
    return None


def _store_last_replay(root: Path, replay_path: Path) -> None:
    state_path = _replay_state_path(root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(str(replay_path), encoding="utf-8")


def _choose_replay_file_macos(root: Path, last_path: Path | None) -> Path | None:
    initialdir = root / "runs" / "smoke" / "replays"
    default_location = None
    if last_path is not None and last_path.exists():
        default_location = last_path
    elif initialdir.exists():
        default_location = initialdir
    script_lines = ['tell application "System Events" to activate', 'set promptText to "Select replay file"']
    if default_location is not None:
        script_lines.append(f"set defaultLoc to POSIX file {json.dumps(str(default_location))} as alias")
        script_lines.append('set chosenFile to choose file with prompt promptText default location defaultLoc')
    else:
        script_lines.append('set chosenFile to choose file with prompt promptText')
    script_lines.append("POSIX path of chosenFile")
    result = subprocess.run(
        ["osascript", "-e", "\n".join(script_lines)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    if not selected:
        return None
    return Path(selected)


def _choose_replay_file(root: Path, last_path: Path | None) -> Path | None:
    if platform.system() == "Darwin":
        return _choose_replay_file_macos(root, last_path)
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("tkinter is required for replay file selection") from exc
    initialdir = root / "runs" / "smoke" / "replays"
    initialfile = None
    if last_path is not None:
        initialdir = last_path.parent
        initialfile = last_path.name
    tk_root = tk.Tk()
    tk_root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select replay file",
        initialdir=str(initialdir),
        initialfile=initialfile,
        filetypes=[("Replay files", "*.json"), ("All files", "*")],
    )
    tk_root.destroy()
    if not selected:
        return None
    return Path(selected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map",
        default="switchback",
        help="Map name (e.g. dev_2lanes) or path to json (ignored when --test is set).",
    )
    ap.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable GUI test mode using test definitions.",
    )
    ap.add_argument(
        "--test",
        default=None,
        help="Test name from data/tests (implies --test-mode).",
    )
    ap.add_argument(
        "--replay",
        action="store_true",
        help="Replay a recorded run (opens file picker).",
    )
    ap.add_argument(
        "--replay-file",
        default=None,
        help="Replay file path (skips picker).",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]

    if args.replay or args.replay_file:
        replay_path = None
        if args.replay_file:
            replay_path = Path(args.replay_file)
        else:
            last_path = _load_last_replay(root)
            replay_path = _choose_replay_file(root, last_path)
        if replay_path is None:
            print("[replay] no file selected")
            return
        _store_last_replay(root, replay_path)
        run_replay(replay_path)
        return

    test_mode = bool(args.test_mode or args.test)
    test_definition = None
    if test_mode:
        test_dir = resolve_tests_dir(root)
        tests = load_test_definitions(test_dir)
        if not tests:
            raise ValueError(f"No tests found in {test_dir}")
        test_name = args.test or sorted(tests)[0]
        if test_name not in tests:
            available = ", ".join(sorted(tests))
            raise ValueError(f"Unknown test {test_name!r}. Available: {available}")
        test_definition = tests[test_name]
        map_path = resolve_map_path(test_definition.map, root)
        run(map_path, test_definition=test_definition)
        return

    map_path = resolve_map_path(args.map, root)
    run(map_path)


if __name__ == "__main__":
    main()
