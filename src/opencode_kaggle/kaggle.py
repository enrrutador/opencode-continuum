"""Kaggle-specific helpers for Dataset identity and environment.

Does NOT treat /kaggle/datasets as a writable filesystem.
Persistence is always: download via kagglehub → local store → upload via kagglehub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_dataset_id(dataset_id: Optional[str] = None) -> str:
    """Resolve Dataset handle from argument or environment.

    Expected format: owner/dataset
    Configuration (not a secret): OPENCODE_CLOUD_DATASET
    """
    did = dataset_id or os.environ.get("OPENCODE_CLOUD_DATASET") or os.environ.get(
        "KAGGLE_DATASET_ID"
    )
    if not did:
        raise RuntimeError(
            "Kaggle dataset not configured. "
            "Set OPENCODE_CLOUD_DATASET to 'owner/dataset'."
        )
    parts = did.strip().split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"OPENCODE_CLOUD_DATASET must be 'owner/dataset', got {did!r}"
        )
    if "versions" in parts:
        raise ValueError("dataset id must not include a version segment")
    return f"{parts[0]}/{parts[1]}"


def working_root() -> Path:
    return Path("/kaggle/working")


def cloud_root() -> Path:
    return working_root() / "opencode_cloud"
