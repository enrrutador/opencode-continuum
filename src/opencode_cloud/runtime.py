"""Runtime detection and path management for ephemeral Kaggle (and local) runtimes.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RuntimePaths:
    """Paths inside an ephemeral runtime."""

    working: Path
    home: Path
    cloud_root: Path
    workspace: Path
    xdg_root: Path
    xdg_data: Path
    xdg_config: Path
    xdg_state: Path
    xdg_cache: Path
    opencode_data: Path
    opencode_config: Path
    checkpoints: Path
    metadata: Path
    logs: Path
    staging: Path


def detect_runtime() -> str:
    if is_kaggle():
        return "kaggle"
    return "local"


def get_paths(runtime: Optional[str] = None) -> RuntimePaths:
    runtime = runtime or detect_runtime()
    if runtime == "kaggle":
        working = Path("/kaggle/working")
        home = Path.home()
    else:
        working = Path(os.environ.get("OPENCODE_CLOUD_WORKDIR", "/tmp/opencode-cloud"))
        home = Path.home()

    cloud_root = working / "opencode_cloud"
    xdg_root = cloud_root / "xdg"
    xdg_data = xdg_root / "share"
    xdg_config = xdg_root / "config"
    xdg_state = xdg_root / "state"
    xdg_cache = xdg_root / "cache"
    return RuntimePaths(
        working=working,
        home=home,
        cloud_root=cloud_root,
        workspace=cloud_root / "workspace",
        xdg_root=xdg_root,
        xdg_data=xdg_data,
        xdg_config=xdg_config,
        xdg_state=xdg_state,
        xdg_cache=xdg_cache,
        opencode_data=xdg_data / "opencode",
        opencode_config=xdg_config,
        checkpoints=cloud_root / "checkpoints",
        metadata=cloud_root / "metadata",
        logs=cloud_root / "logs",
        staging=cloud_root / "staging",
    )


def ensure_dirs(paths: RuntimePaths) -> None:
    for p in (
        paths.cloud_root,
        paths.workspace,
        paths.xdg_root,
        paths.xdg_data,
        paths.xdg_config,
        paths.xdg_state,
        paths.xdg_cache,
        paths.opencode_data,
        paths.opencode_config,
        paths.checkpoints,
        paths.metadata,
        paths.logs,
        paths.staging,
    ):
        p.mkdir(parents=True, exist_ok=True)


def is_kaggle() -> bool:
    return (
        os.path.exists("/kaggle")
        or bool(os.environ.get("KAGGLE_KERNEL_EXECUTION"))
        or bool(os.environ.get("KAGGLE_CONTAINER_TYPE"))
    )


def is_linux() -> bool:
    return platform.system().lower() == "linux"
