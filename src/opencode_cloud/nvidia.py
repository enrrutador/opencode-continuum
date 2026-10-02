"""NVIDIA NIM provider configuration.

API key is never written to config, README, Dataset, logs, argv, or temp files.
Always referenced as {env:NVIDIA_API_KEY}.

HTTP calls use urllib with the Authorization header kept only in memory.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def fetch_models(api_key: str) -> list[str]:
    """Fetch available NVIDIA NIM models via Python HTTP.

    The API key stays in memory only (Authorization header).
    It is never placed in subprocess argv, command lines, or files.
    Returns empty list on failure (caller continues without model).
    """
    if not api_key:
        return []
    req = urllib.request.Request(
        f"{NVIDIA_BASE_URL}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "opencode-continuum/5.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        return [item["id"] for item in payload.get("data", []) if item.get("id")]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError):
        return []
    except Exception:
        # Never re-raise with api_key in the message
        return []


def select_model(available: list[str], preferred: str = "") -> Optional[str]:
    """Select model: preferred → first nemotron → first available."""
    if preferred and preferred in available:
        return preferred
    nemotron = [m for m in available if "nemotron" in m.lower()]
    if nemotron:
        return nemotron[0]
    return available[0] if available else None


def build_provider_config(
    api_key_env: str = "NVIDIA_API_KEY", model: Optional[str] = None
) -> dict:
    """Build the NVIDIA provider section. Key is env reference only."""
    cfg = {
        "name": "NVIDIA NIM",
        "npm": "@ai-sdk/openai-compatible",
        "options": {
            "baseURL": NVIDIA_BASE_URL,
            "apiKey": f"{{env:{api_key_env}}}",
        },
        "models": {},
    }
    if model:
        cfg["models"][model] = {"name": model}
    return {"nvidia": cfg}


def build_opencode_config(
    model: Optional[str] = None, permission: str = "allow"
) -> dict:
    """Build a complete OpenCode configuration dict."""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "permission": permission,
        "compaction": {"auto": True, "prune": False},
        "provider": build_provider_config(model=model),
    }
    if model:
        cfg["model"] = f"nvidia/{model}"
    return cfg
