"""GitHub synchronization for OpenCode Cloud Workstation.

Security-first:
- subprocess with list arguments only (never uses shell mode)
- Temporary credential file, always deleted in finally
- Token never persisted in workspace or Dataset
- GitHub is code versioning, NOT primary workstation persistence
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


_URL_RE = re.compile(r"^https://github\.com/[^/]+/[^/]+(\.git)?$")


def validate_repo_url(url: str) -> str:
    """Validate and normalize a GitHub HTTPS URL."""
    if not url:
        return ""
    url = url.strip().rstrip("/")
    if not url.endswith(".git"):
        url += ".git"
    if not _URL_RE.match(url):
        raise ValueError(f"Invalid or unsupported GitHub repository URL: {url!r}")
    return url


def _run(
    args: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=capture,
        text=True,
    )


def init_repo(workspace: Path) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        _run(["git", "init"], cwd=workspace, capture=False)
        _run(["git", "checkout", "-B", "main"], cwd=workspace, capture=False)
    _run(
        ["git", "config", "user.name", "OpenCode Cloud Workstation"],
        cwd=workspace,
        capture=False,
    )
    _run(
        ["git", "config", "user.email", "opencode-cloud@localhost"],
        cwd=workspace,
        capture=False,
    )


def configure_remote(workspace: Path, repo_url: str) -> None:
    repo_url = validate_repo_url(repo_url)
    workspace = Path(workspace)
    _run(["git", "remote", "remove", "origin"], cwd=workspace, capture=True)
    _run(["git", "remote", "add", "origin", repo_url], cwd=workspace, capture=True)


@contextmanager
def temporary_credential_helper(
    token: str, workspace: Path
) -> Generator[Path, None, None]:
    """Create a temp credential file, configure helper, yield, then always delete."""
    workspace = Path(workspace)
    cred_file: Optional[Path] = None
    try:
        # Use system temp — NOT inside workspace (so it never enters Dataset)
        fd, name = tempfile.mkstemp(prefix="git-cred-", suffix=".tmp")
        os.close(fd)
        cred_file = Path(name)
        cred_file.write_text(
            f"https://x-access-token:{token}@github.com\n", encoding="utf-8"
        )
        try:
            os.chmod(cred_file, 0o600)
        except Exception:
            pass
        _run(
            ["git", "config", "credential.helper", f"store --file={cred_file}"],
            cwd=workspace,
            capture=True,
        )
        yield cred_file
    finally:
        if cred_file is not None and cred_file.exists():
            try:
                cred_file.unlink()
            except Exception:
                pass
        _run(
            ["git", "config", "--unset", "credential.helper"],
            cwd=workspace,
            capture=True,
        )


def fetch_origin(workspace: Path, env: Optional[dict] = None) -> bool:
    result = _run(["git", "fetch", "origin"], cwd=workspace, env=env)
    return result.returncode == 0


def merge_origin_main(workspace: Path, env: Optional[dict] = None) -> bool:
    result = _run(
        ["git", "merge", "origin/main", "--no-edit"], cwd=workspace, env=env
    )
    return result.returncode == 0


def add_all(workspace: Path) -> None:
    _run(["git", "add", "-A"], cwd=workspace, capture=True)


def status_porcelain(workspace: Path) -> str:
    result = _run(["git", "status", "--porcelain"], cwd=workspace)
    return result.stdout or ""


def commit(workspace: Path, message: str) -> bool:
    result = _run(["git", "commit", "-m", message], cwd=workspace)
    return result.returncode == 0


def push_origin_main(
    workspace: Path, token: Optional[str] = None, env: Optional[dict] = None
) -> bool:
    """Push using temporary credential helper if token provided."""
    workspace = Path(workspace)
    if token:
        try:
            with temporary_credential_helper(token, workspace):
                result = _run(
                    ["git", "push", "-u", "origin", "main"],
                    cwd=workspace,
                    env=env,
                )
            return result.returncode == 0
        except Exception:
            return False
    result = _run(
        ["git", "push", "-u", "origin", "main"], cwd=workspace, env=env
    )
    return result.returncode == 0


def sync_to_remote(
    workspace: Path,
    message: str,
    token: Optional[str] = None,
    env: Optional[dict] = None,
) -> bool:
    """Add, commit, and push. Token file is always cleaned up."""
    add_all(workspace)
    status = status_porcelain(workspace)
    if status.strip():
        commit(workspace, message)
    if token:
        return push_origin_main(workspace, token=token, env=env)
    return True
