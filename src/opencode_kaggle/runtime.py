"""Kaggle-specific runtime detection and documentation helpers."""

from __future__ import annotations

import os
from pathlib import Path


def is_kaggle_runtime() -> bool:
    return (
        os.path.exists("/kaggle")
        or bool(os.environ.get("KAGGLE_KERNEL_EXECUTION"))
        or bool(os.environ.get("KAGGLE_CONTAINER_TYPE"))
    )


def get_runtime_paths() -> dict:
    working = Path("/kaggle/working")
    return {
        "working": working,
        "input": Path("/kaggle/input"),
        "datasets_readonly_hint": Path("/kaggle/datasets"),
        "home": Path.home(),
        "cloud_root": working / "opencode_cloud",
    }


def describe_opencode_web_access(port: int = 4096) -> str:
    """Short description; prefer AccessInfo from KaggleProxyAccess.resolve()."""
    return (
        f"OpenCode listens on 127.0.0.1:{port}. "
        "External access uses Kaggle Jupyter Proxy "
        "(kkb-production.jupyter-proxy.kaggle.net/.../proxy/proxy/<PORT>). "
        "bootstrap() builds the URL from list_running_servers() and HTTP-probes it "
        "before marking available."
    )
