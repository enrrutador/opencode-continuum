"""Checkpoint strategy for OpenCode Cloud Workstation.

Two levels:

1. Local checkpoint (cheap, frequent) under /kaggle/working/opencode_cloud/
2. Remote checkpoint (policy-gated) → Kaggle Dataset only (never GitHub)

CheckpointManager decides WHETHER to publish.
CheckpointScheduler runs the periodic observe → local → remote loop.
KagglePersistence executes the publish.
Watchdog does NOT publish.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Policy defaults (RPO target under normal operation)
LOCAL_CHECKPOINT_INTERVAL = 60  # seconds between local observations/saves
REMOTE_CHECKPOINT_MIN_INTERVAL = 900  # 15 minutes between normal remote publishes
MIN_PUBLISH_INTERVAL_SECONDS = REMOTE_CHECKPOINT_MIN_INTERVAL  # alias


class PublishReason(str, Enum):
    NONE = "none"
    COOLDOWN_AND_CHANGES = "cooldown_and_changes"
    SHUTDOWN = "shutdown"
    EXPLICIT = "explicit"
    RECOVERY = "recovery"


@dataclass
class CheckpointPolicy:
    local_interval: int = LOCAL_CHECKPOINT_INTERVAL
    min_publish_interval: int = REMOTE_CHECKPOINT_MIN_INTERVAL
    always_on_shutdown: bool = True
    always_on_explicit: bool = True
    always_on_recovery: bool = False
    max_fingerprint_files: int = 50_000  # hard cap; overflow → incomplete


@dataclass
class CheckpointState:
    last_local_checkpoint: float = 0.0
    last_remote_publish: float = 0.0
    remote_publish_count: int = 0
    pending_significant_changes: int = 0
    remote_pending: bool = False
    last_workspace_fingerprint: str = ""
    fingerprint_incomplete: bool = False


@dataclass
class FingerprintResult:
    digest: str
    file_count: int
    incomplete: bool
    truncated_at: int = 0


def workspace_fingerprint(
    workspace: Path,
    *,
    max_files: int = 50_000,
) -> FingerprintResult:
    """Fingerprint workspace using path + size + mtime_ns for every regular file.

    Detects: create, delete, rename, size change, content rewrite (via mtime_ns).
    If file count exceeds max_files, marks incomplete=True (never silently
    treats a partial scan as a full representation).
    """
    workspace = Path(workspace)
    if not workspace.exists():
        return FingerprintResult(digest="empty", file_count=0, incomplete=False)

    entries: list[str] = []
    truncated = False
    try:
        for p in sorted(workspace.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            try:
                st = p.stat()
                rel = p.relative_to(workspace).as_posix()
                mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                entries.append(f"{rel}:{st.st_size}:{mtime_ns}")
            except OSError:
                continue
            if len(entries) >= max_files:
                truncated = True
                break
    except OSError:
        return FingerprintResult(digest="unreadable", file_count=0, incomplete=True)

    raw = "\n".join(entries).encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw).hexdigest()
    if truncated:
        digest = hashlib.sha256((digest + f":truncated:{max_files}").encode()).hexdigest()
    return FingerprintResult(
        digest=digest,
        file_count=len(entries),
        incomplete=truncated,
        truncated_at=max_files if truncated else 0,
    )


class CheckpointManager:
    """Decides when to perform local checkpoints and remote publishes."""

    def __init__(self, policy: Optional[CheckpointPolicy] = None):
        self.policy = policy or CheckpointPolicy()
        self.state = CheckpointState()

    def record_local_checkpoint(self) -> None:
        self.state.last_local_checkpoint = time.time()

    def mark_significant_change(self) -> None:
        self.state.pending_significant_changes += 1
        self.state.remote_pending = True

    def observe_workspace(self, workspace: Path) -> bool:
        """Compare fingerprint; return True if significant change recorded."""
        return self.observe_paths(workspace)

    def observe_paths(self, *paths: Path) -> bool:
        import hashlib
        h = hashlib.sha256()
        incomplete = False
        for root in paths:
            root = Path(root)
            if not root.exists():
                h.update(b"missing:")
                h.update(str(root).encode())
                continue
            result = workspace_fingerprint(root, max_files=self.policy.max_fingerprint_files)
            incomplete = incomplete or result.incomplete
            h.update(result.digest.encode())
        digest = h.hexdigest()
        self.state.fingerprint_incomplete = incomplete
        if not self.state.last_workspace_fingerprint:
            self.state.last_workspace_fingerprint = digest
            return False
        if digest != self.state.last_workspace_fingerprint:
            self.state.last_workspace_fingerprint = digest
            self.mark_significant_change()
            return True
        return False

    def set_baseline_fingerprint(self, workspace: Path) -> str:
        import hashlib

        result = workspace_fingerprint(
            workspace, max_files=self.policy.max_fingerprint_files
        )
        # Consistente con observe_paths(): hash del digest, no digest crudo.
        # Si no, baseline nunca iguala y siempre detecta "changed".
        h = hashlib.sha256()
        h.update(result.digest.encode())
        digest = h.hexdigest()
        self.state.last_workspace_fingerprint = digest
        self.state.fingerprint_incomplete = result.incomplete
        return digest

    def should_publish_remote(
        self,
        *,
        reason: Optional[PublishReason] = None,
        significant_changes: Optional[int] = None,
        now: Optional[float] = None,
    ) -> tuple[bool, PublishReason]:
        clock = now if now is not None else time.time()
        changes = (
            significant_changes
            if significant_changes is not None
            else self.state.pending_significant_changes
        )

        if reason == PublishReason.EXPLICIT and self.policy.always_on_explicit:
            return True, PublishReason.EXPLICIT
        if reason == PublishReason.SHUTDOWN and self.policy.always_on_shutdown:
            return True, PublishReason.SHUTDOWN
        if reason == PublishReason.RECOVERY and self.policy.always_on_recovery:
            return True, PublishReason.RECOVERY

        elapsed = clock - self.state.last_remote_publish
        if changes >= 1 and elapsed >= self.policy.min_publish_interval:
            return True, PublishReason.COOLDOWN_AND_CHANGES
        if self.state.remote_pending and elapsed >= self.policy.min_publish_interval:
            return True, PublishReason.COOLDOWN_AND_CHANGES
        return False, PublishReason.NONE

    def record_remote_publish(self, *, now: Optional[float] = None) -> None:
        clock = now if now is not None else time.time()
        self.state.last_remote_publish = clock
        self.state.remote_publish_count += 1
        self.state.pending_significant_changes = 0
        self.state.remote_pending = False

    def get_state(self) -> dict:
        return {
            "last_local_checkpoint": self.state.last_local_checkpoint,
            "last_remote_publish": self.state.last_remote_publish,
            "remote_publish_count": self.state.remote_publish_count,
            "pending_significant_changes": self.state.pending_significant_changes,
            "remote_pending": self.state.remote_pending,
            "min_publish_interval": self.policy.min_publish_interval,
            "local_interval": self.policy.local_interval,
            "fingerprint_incomplete": self.state.fingerprint_incomplete,
            "last_workspace_fingerprint": self.state.last_workspace_fingerprint[:16]
            if self.state.last_workspace_fingerprint
            else "",
        }
