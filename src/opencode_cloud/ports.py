"""Central port configuration for OpenCode Cloud Workstation."""

from __future__ import annotations

import os

DEFAULT_OPENCODE_PORT = 4096
DEFAULT_ENTRY_REDIRECTOR_PORT = 4097


def get_opencode_port() -> int:
    raw = os.environ.get("OPENCODE_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_OPENCODE_PORT


def get_entry_redirector_port() -> int:
    raw = os.environ.get("OPENCODE_REDIRECTOR_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_ENTRY_REDIRECTOR_PORT
