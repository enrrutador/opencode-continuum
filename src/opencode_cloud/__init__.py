"""opencode-continuum — platform-agnostic core.

Persistent OpenCode workstation for ephemeral cloud runtimes.
"""

__version__ = "5.0.0"

from .checkpoint import CheckpointManager, CheckpointPolicy, PublishReason
from .persistence import (
    KagglePersistence,
    PersistentStore,
    RecoveryResult,
    RecoveryStatus,
)

__all__ = [
    "CheckpointManager",
    "CheckpointPolicy",
    "PublishReason",
    "KagglePersistence",
    "PersistentStore",
    "RecoveryResult",
    "RecoveryStatus",
    "__version__",
]
