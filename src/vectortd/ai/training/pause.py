from __future__ import annotations

import contextlib
import select
import sys
import threading
from typing import TextIO

try:
    import termios
except Exception:  # pragma: no cover - platform guard
    termios = None


_SPECIAL_KEYS = {
    "space": " ",
    "tab": "\t",
    "enter": "\n",
    "return": "\n",
    "esc": "\x1b",
    "escape": "\x1b",
}


def normalize_pause_key(value: str | None) -> str:
    if value is None:
        return " "
    cleaned = value.strip()
    if not cleaned:
        return " "
    lowered = cleaned.lower()
    if lowered in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[lowered]
    return cleaned[0]


def _format_key(value: str) -> str:
    if value == " ":
        return "Space"
    if value == "\t":
        return "Tab"
    if value == "\n":
        return "Enter"
    if value == "\x1b":
        return "Esc"
    return value


class PauseController:
    def __init__(
        self,
        *,
        enabled: bool,
        key: str,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.enabled = enabled
        self.key = key
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._running = threading.Event()
        self._running.set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._term_state: list[int] | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None:
            return
        if termios is None or not self._is_tty():
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._listen, name="pause-listener", daemon=True)
        self._thread.start()
        self._log(f"pause=ready key={_format_key(self.key)}")

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        self._running.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def wait_if_paused(self) -> None:
        if not self.enabled:
            return
        self._running.wait()

    def toggle(self) -> None:
        if self._running.is_set():
            self._running.clear()
            self._log("pause=on")
        else:
            self._running.set()
            self._log("pause=off")

    def _is_tty(self) -> bool:
        return bool(getattr(self._stdin, "isatty", lambda: False)())

    def _log(self, message: str) -> None:
        if self._stdout is None:
            return
        with contextlib.suppress(Exception):
            self._stdout.write(message + "\n")
            self._stdout.flush()

    def _listen(self) -> None:
        if termios is None:
            return
        fd = self._stdin.fileno()
        try:
            self._term_state = termios.tcgetattr(fd)
            new_state = termios.tcgetattr(fd)
            new_state[3] = new_state[3] & ~termios.ICANON
            new_state[3] = new_state[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_state)
            while not self._stop.is_set():
                ready, _, _ = select.select([self._stdin], [], [], 0.2)
                if not ready:
                    continue
                ch = self._stdin.read(1)
                if not ch:
                    continue
                if ch == self.key:
                    self.toggle()
        finally:
            if self._term_state is not None:
                with contextlib.suppress(Exception):
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._term_state)
