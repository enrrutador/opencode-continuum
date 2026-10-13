"""Cloudflare Tunnel access for the Kaggle OpenCode workstation.

Primary access layer:
  1. Named Cloudflare Tunnel when CLOUDFLARE_TUNNEL_TOKEN is available.
  2. Cloudflare Quick Tunnel otherwise.

The tunnel URL is ephemeral and must never be persisted.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from .access import AccessInfo

CLOUDFLARED_VERSION = "2025.2.1"


def _persistent_bin_dir() -> Path:
    """Dir persistente en /kaggle/working (sobrevive restart de kernel).

    Patrón tomado de kagglevstunnel / vscode-tunnel: guardar binarios
    en /kaggle/working en vez de $HOME (efímero). Si no estamos en Kaggle,
    cae a ~/.local/bin.
    """
    for cand in (
        Path("/kaggle/working/opencode_cloud/bin"),
        Path(os.environ.get("OPENCODE_CLOUD_WORKDIR", "")) / "opencode_cloud" / "bin"
        if os.environ.get("OPENCODE_CLOUD_WORKDIR")
        else None,
    ):
        if cand is None:
            continue
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except OSError:
            continue
    return Path.home() / ".local" / "bin"

_URL_RE = re.compile(
    r"https://[a-zA-Z0-9.-]+\.(?:trycloudflare\.com|cfargotunnel\.com)[^\s]*"
)


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)

    return False


def _download_cloudflared(url: str, *, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "opencode-continuum/5.0"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def ensure_cloudflared(bin_dir: Optional[Path] = None) -> str:
    # 0. Override explícito (tests / custom).
    override = os.environ.get("OPENCODE_CLOUDFLARED_BIN", "").strip()
    if override:
        p = Path(override)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    existing = shutil.which("cloudflared")

    if existing:
        return existing

    # 1. Reuso persistente: /kaggle/working primero (sobrevive restart),
    #    luego el dir pedido, luego legacy ~/.local/bin.
    candidates: list[Path] = []
    if bin_dir is not None:
        candidates.append(Path(bin_dir))
    try:
        candidates.append(_persistent_bin_dir())
    except Exception:
        pass
    candidates.append(Path.home() / ".local" / "bin")

    for cand in candidates:
        try:
            target = cand / "cloudflared"
            if target.is_file() and os.access(target, os.X_OK):
                return str(target)
        except OSError:
            continue

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "linux":
        raise RuntimeError(
            f"cloudflared auto-install supports Linux only: {system}"
        )

    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        raise RuntimeError(
            f"Unsupported architecture: {machine}"
        )

    pinned_url = (
        "https://github.com/cloudflare/cloudflared/releases/download/"
        f"{CLOUDFLARED_VERSION}/cloudflared-linux-{arch}"
    )
    latest_url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{arch}"
    )

    data: Optional[bytes] = None
    last_error: Optional[Exception] = None
    for url in (pinned_url, latest_url):
        try:
            data = _download_cloudflared(url)
            if len(data) < 1_000_000:
                raise RuntimeError("cloudflared download too small")
            break
        except Exception as exc:  # pin 404 -> probar latest
            last_error = exc
            data = None
            continue
    if data is None:
        raise RuntimeError(
            f"cloudflared download failed: {type(last_error).__name__ if last_error else 'unknown'}"
        )

    # Destino: primer candidato escribible (persistente primero).
    target: Optional[Path] = None
    dest_dir: Optional[Path] = None
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            target = cand / "cloudflared"
            dest_dir = cand
            break
        except OSError:
            continue
    assert target is not None and dest_dir is not None

    fd, tmp_name = tempfile.mkstemp(
        prefix="cloudflared_",
        dir=str(dest_dir),
    )

    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_bytes(data)
        tmp_path.chmod(0o755)
        tmp_path.replace(target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return str(target)


def parse_tunnel_url(text: str) -> Optional[str]:
    matches = _URL_RE.findall(text)

    for match in matches:
        if (
            "trycloudflare.com" in match
            or "cfargotunnel.com" in match
        ):
            return match.rstrip(".,;)")

    return matches[0].rstrip(".,;)") if matches else None


class CloudflareAccess:

    def __init__(
        self,
        port: int = 4096,
        *,
        tunnel_token: Optional[str] = None,
        cloudflared_bin: Optional[str] = None,
        log_path: Optional[Path] = None,
    ):
        self.port = int(port)

        self.tunnel_token = (
            tunnel_token
            or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
        )

        self.cloudflared_bin = cloudflared_bin
        self.log_path = (
            Path(log_path)
            if log_path
            else None
        )

        self._proc = None
        self._url = None
        self._output = []
        self._reader = None

    def start(self, wait_url_timeout: float = 45.0) -> AccessInfo:

        if not wait_for_port(
            "127.0.0.1",
            self.port,
            timeout=5.0,
        ):
            return AccessInfo(
                available=False,
                status="opencode_not_listening",
                message=(
                    f"OpenCode is not listening on "
                    f"127.0.0.1:{self.port}"
                ),
                local_port=self.port,
                opencode_listening=False,
                provider="cloudflare",
            )

        try:
            bin_path = (
                self.cloudflared_bin
                or ensure_cloudflared()
            )

        except Exception as exc:
            return AccessInfo(
                available=False,
                status="cloudflared_unavailable",
                message=(
                    "Could not obtain cloudflared: "
                    f"{type(exc).__name__}"
                ),
                local_port=self.port,
                opencode_listening=True,
                provider="cloudflare",
            )

        if self.tunnel_token:
            cmd = [
                bin_path,
                "tunnel",
                "run",
                "--token",
                self.tunnel_token,
            ]
            provider = "cloudflare_named_tunnel"
        else:
            cmd = [
                bin_path,
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{self.port}",
            ]
            provider = "cloudflare_quick_tunnel"

        log_handle = None

        if self.log_path:
            self.log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            log_handle = open(
                self.log_path,
                "a",
                encoding="utf-8",
            )

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

        except Exception as exc:
            if log_handle:
                log_handle.close()

            return AccessInfo(
                available=False,
                status="tunnel_start_failed",
                message=(
                    "Failed to start cloudflared: "
                    f"{type(exc).__name__}"
                ),
                local_port=self.port,
                opencode_listening=True,
                provider=provider,
            )

        def reader():
            stream = self._proc.stdout

            if stream is None:
                return

            try:
                for line in stream:

                    safe = line

                    # Never retain tunnel credentials.
                    if (
                        "token" in line.lower()
                        and len(line) > 40
                    ):
                        safe = "[redacted tunnel line]\n"

                    self._output.append(safe)

                    if len(self._output) > 200:
                        self._output = self._output[-100:]

                    if log_handle:
                        log_handle.write(safe)
                        log_handle.flush()

            except Exception:
                pass

            finally:
                if log_handle:
                    try:
                        log_handle.close()
                    except Exception:
                        pass

        self._reader = threading.Thread(
            target=reader,
            daemon=True,
        )
        self._reader.start()

        deadline = time.time() + wait_url_timeout
        url = None

        while time.time() < deadline:

            if self._proc.poll() is not None:
                return AccessInfo(
                    available=False,
                    status="tunnel_exited",
                    message=(
                        "cloudflared exited before "
                        "the URL was available"
                    ),
                    local_port=self.port,
                    opencode_listening=True,
                    provider=provider,
                )

            url = parse_tunnel_url(
                "\n".join(self._output[-50:])
            )

            if url:
                break

            time.sleep(0.4)

        self._url = url

        # Named tunnel can be healthy without emitting
        # a public URL because the hostname is configured
        # externally in Cloudflare.
        if self.tunnel_token and not url:
            return AccessInfo(
                available=True,
                url=None,
                status="named_tunnel_running",
                message=(
                    "Cloudflare Named Tunnel is running. "
                    "Its configured hostname is external "
                    "to this runtime."
                ),
                local_port=self.port,
                opencode_listening=True,
                provider=provider,
                authentication="opencode_basic_auth",
                proxy_reachable=True,
            )

        if not url:
            return AccessInfo(
                available=False,
                status="cloudflare_url_timeout",
                message=(
                    "Timed out waiting for Cloudflare "
                    "Tunnel URL"
                ),
                local_port=self.port,
                opencode_listening=True,
                provider=provider,
            )

        return AccessInfo(
            available=True,
            url=url,
            url_redacted=url,  # CF quick tunnel URLs have no session JWT
            status="ready",
            message=(
                "Cloudflare Tunnel is ready. "
                "URL is session-scoped and must not "
                "be persisted."
            ),
            local_port=self.port,
            opencode_listening=True,
            provider=provider,
            proxy_reachable=True,
            authentication=(
                "opencode_basic_auth"
                if os.environ.get(
                    "OPENCODE_SERVER_PASSWORD"
                )
                else "none"
            ),
        )

    def stop(self):
        if self._proc is None:
            return

        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)

        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

        self._proc = None

    def status(self):
        alive = (
            self._proc is not None
            and self._proc.poll() is None
        )

        return {
            "running": alive,
            "url": self._url,
            "pid": (
                self._proc.pid
                if self._proc and alive
                else None
            ),
        }
