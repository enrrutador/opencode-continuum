"""OpenCode bootstrap and management.

Idempotent:
- ensure_node(): if Node exists → skip; else install via explicit local script
- ensure_opencode(): if OpenCode exists → skip; else install via npm

Config is merged incrementally; NVIDIA provider is preserved/updated without
wiping unrelated provider blocks when possible.
Secrets are never written into the config file — only env references.

Security:
- No shell flag on subprocess
- No shell string invocation
- No remote pipe-to-shell install
- NodeSource setup is downloaded to a temp file, then executed as a local file
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

NODE_SETUP_URL = "https://deb.nodesource.com/setup_22.x"


def ensure_node() -> str:
    """Ensure Node.js is available. Idempotent. Returns version string.

    Installation strategy (in order):
    1. Use existing node binary if present
    2. Download NodeSource setup script to a temp file (HTTP status checked)
    3. Execute that local file with explicit argv (no pipe to shell)
    4. apt-get install nodejs with explicit argv
    5. Delete the temp script
    """
    result = subprocess.run(
        ["node", "--version"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()

    # Prefer preinstalled node on Kaggle when available; only then install.
    setup_path: Optional[Path] = None
    try:
        # Download setup script explicitly — never remote-pipe install
        req = urllib.request.Request(
            NODE_SETUP_URL,
            headers={"User-Agent": "opencode-continuum/5.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if getattr(resp, "status", 200) not in (200, 301, 302):
                raise RuntimeError(f"NodeSource HTTP status {getattr(resp, 'status', '?')}")
            body = resp.read()

        fd, tmp_name = tempfile.mkstemp(suffix=".sh", prefix="nodesource_")
        os.close(fd)
        setup_path = Path(tmp_name)
        setup_path.write_bytes(body)
        setup_path.chmod(0o700)

        # Execute local file with explicit arguments — not a remote pipe
        subprocess.run(
            ["bash", str(setup_path)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "nodejs"],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except Exception as e:
        raise RuntimeError(
            "Node.js is required but not installed and auto-install failed. "
            "Install Node.js manually (e.g. apt-get install -y nodejs). "
            f"Error: {type(e).__name__}"
        ) from e
    finally:
        if setup_path is not None and setup_path.exists():
            try:
                setup_path.unlink()
            except OSError:
                pass

    result = subprocess.run(
        ["node", "--version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("Node.js installation failed")
    return result.stdout.strip()


def ensure_opencode() -> str:
    """Ensure OpenCode CLI is installed. Idempotent. Returns binary path."""
    bin_path = shutil.which("opencode")
    if bin_path:
        return bin_path

    subprocess.run(
        ["npm", "install", "-g", "opencode-ai"],
        check=True,
        capture_output=True,
        timeout=300,
    )
    bin_path = shutil.which("opencode")
    if not bin_path:
        raise RuntimeError("OpenCode installation failed")
    return bin_path


def write_opencode_config(
    config_path: Path,
    model: Optional[str] = None,
    permission: str = "allow",
) -> Path:
    """Write or merge OpenCode configuration incrementally."""
    from .nvidia import build_opencode_config, build_provider_config

    new_cfg = build_opencode_config(model=model, permission=permission)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    if existing:
        merged = dict(existing)
        merged["$schema"] = new_cfg["$schema"]
        merged["permission"] = new_cfg["permission"]
        merged["compaction"] = new_cfg["compaction"]
        providers = dict(merged.get("provider") or {})
        nvidia_block = build_provider_config(model=model)["nvidia"]
        providers["nvidia"] = nvidia_block
        merged["provider"] = providers
        if model:
            merged["model"] = new_cfg["model"]
        final_cfg = merged
    else:
        final_cfg = new_cfg

    config_path.write_text(
        json.dumps(final_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config_path


def start_opencode_web(
    opencode_bin: str,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 4096,
    env: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    """Start OpenCode Web server. Returns Popen. Caller monitors it."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = workspace.parent / "logs" / "opencode-web.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [opencode_bin, "web", "--hostname", host, "--port", str(port)]
    log_handle = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return proc
