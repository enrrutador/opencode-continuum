"""Process watchdog for OpenCode.

Responsibility (and only this):
  - Monitor whether the OpenCode process is alive
  - If it dies: detect, call a REAL restart function, log the result

NOT responsible for:
  - Persistence / Dataset publish
  - GitHub commits
  - Checkpoint policy
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class Watchdog:
    """Monitors an OpenCode process and restarts it on failure."""

    def __init__(
        self,
        check_interval: float = 30.0,
        restart_fn: Optional[Callable[[], object]] = None,
        process_poll: Optional[Callable[[], Optional[int]]] = None,
        health_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Args:
            restart_fn: Callable that starts a new OpenCode process and returns
                        an object with a .poll() method (e.g. subprocess.Popen).
                        MUST be a real restart, not a no-op.
            process_poll: Optional alternate way to poll process liveness.
                          If not set, uses the last process returned by restart_fn.
            health_check: Optional callback that returns True if the OpenCode
                          service is actually reachable (e.g. TCP port open).
                          When set, it takes precedence over process_poll.
        """
        self.check_interval = check_interval
        self.restart_fn = restart_fn
        self._process_poll = process_poll
        self._health_check = health_check
        self._process = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.restart_count = 0
        self.last_restart_ok: Optional[bool] = None
        self.last_error: Optional[str] = None

    def set_process(self, proc) -> None:
        self._process = proc

    def set_restart_fn(self, fn: Callable[[], object]) -> None:
        self.restart_fn = fn

    def set_health_check(self, fn: Callable[[], bool]) -> None:
        """Set a health-check callback that returns True if the service is reachable."""
        self._health_check = fn

    def start(self) -> None:
        if self._running:
            return
        if self.restart_fn is None:
            raise RuntimeError(
                "Watchdog requires a real restart_fn; cannot start with no-op"
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Idempotent: safe to call more than once."""
        self._running = False
        thr = self._thread
        if thr is not None and thr.is_alive() and thr is not threading.current_thread():
            thr.join(timeout=5)

    def _is_dead(self) -> bool:
        if self._health_check is not None:
            try:
                return not bool(self._health_check())
            except Exception as e:
                self.last_error = repr(e)
                return True
        if self._process_poll is not None:
            return self._process_poll() is not None
        if self._process is None:
            return True
        poll = getattr(self._process, "poll", None)
        if poll is None:
            return False
        return poll() is not None

    def _loop(self) -> None:
        while self._running:
            try:
                time.sleep(self.check_interval)
                if not self._running:
                    break
                if self._is_dead():
                    self._do_restart()
            except Exception as e:
                self.last_error = repr(e)

    def _do_restart(self) -> None:
        if self.restart_fn is None:
            self.last_restart_ok = False
            self.last_error = "no restart_fn"
            return
        try:
            new_proc = self.restart_fn()
            self._process = new_proc
            self.restart_count += 1
            self.last_restart_ok = True
            self.last_error = None
        except Exception as e:
            self.last_restart_ok = False
            self.last_error = repr(e)

    def status(self) -> dict:
        return {
            "running": self._running,
            "restart_count": self.restart_count,
            "last_restart_ok": self.last_restart_ok,
            "last_error": self.last_error,
            "process_dead": self._is_dead() if self._process or self._process_poll else None,
        }
