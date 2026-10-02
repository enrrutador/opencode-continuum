"""Central port configuration for OpenCode Cloud Workstation."""

from __future__ import annotations

import os

DEFAULT_OPENCODE_PORT = 4096


def get_opencode_port() -> int:
    raw = os.environ.get("OPENCODE_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_OPENCODE_PORT
