"""Periodic checkpoint scheduler — independent of Watchdog.

Bootstrap
 ├── OpenCode process
 ├── Watchdog          → restart OpenCode only
 └── CheckpointScheduler → observe workspace → local/remote checkpoints

Concurrency: a single threading.Lock serializes all checkpoint operations
(scheduler ticks, explicit requests, shutdown final checkpoint).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .checkpoint import CheckpointManager, PublishReason


class CheckpointScheduler:
    """Background worker that drives automatic local/remote checkpoints."""

    def __init__(
        self,
        manager: CheckpointManager,
        *,
        observe_fn: Callable[[], bool],
        local_fn: Callable[[], object],
        remote_fn: Callable[[PublishReason], object],
        lock: Optional[threading.Lock] = None,
        interval: Optional[float] = None,
    ):
        self.manager = manager
        self.observe_fn = observe_fn
        self.local_fn = local_fn
        self.remote_fn = remote_fn
        self.lock = lock or threading.Lock()
        self.interval = float(
            interval if interval is not None else manager.policy.local_interval
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.tick_count = 0
        self.last_error: Optional[str] = None
        self.last_local_result: Optional[object] = None
        self.last_remote_result: Optional[object] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="checkpoint-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def run_once(self, *, reason: Optional[PublishReason] = None) -> dict:
        """Execute one observe → local → maybe remote cycle under the lock."""
        with self.lock:
            return self._tick(reason=reason)

    def shutdown_checkpoint(self) -> dict:
        """Stop loop and perform final remote checkpoint under the lock."""
        self._running = False
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10.0)
        self._thread = None
        with self.lock:
            return self._tick(reason=PublishReason.SHUTDOWN, force_local=True)

    def _loop(self) -> None:
        while self._running:
            try:
                time.sleep(self.interval)
                if not self._running:
                    break
                with self.lock:
                    if not self._running:
                        break
                    self._tick()
            except Exception as e:
                self.last_error = repr(e)

    def _tick(
        self,
        *,
        reason: Optional[PublishReason] = None,
        force_local: bool = False,
    ) -> dict:
        self.tick_count += 1
        changed = False
        try:
            changed = bool(self.observe_fn())
        except Exception as e:
            self.last_error = f"observe: {e!r}"

        local_result = None
        if changed or force_local or reason in (
            PublishReason.EXPLICIT,
            PublishReason.SHUTDOWN,
        ):
            try:
                local_result = self.local_fn()
                self.manager.record_local_checkpoint()
                self.last_local_result = local_result
            except Exception as e:
                self.last_error = f"local: {e!r}"

        should, decided = self.manager.should_publish_remote(reason=reason)
        if reason in (PublishReason.EXPLICIT, PublishReason.SHUTDOWN):
            should = True
            decided = reason

        remote_result = None
        if should:
            try:
                remote_result = self.remote_fn(decided)
                self.last_remote_result = remote_result
            except Exception as e:
                self.last_error = f"remote: {e!r}"

        return {
            "tick": self.tick_count,
            "changed": changed,
            "local": local_result,
            "should_publish": should,
            "reason": decided.value if should else PublishReason.NONE.value,
            "remote": remote_result,
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "tick_count": self.tick_count,
            "interval": self.interval,
            "last_error": self.last_error,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }
