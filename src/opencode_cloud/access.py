"""Kaggle Jupyter Proxy access for OpenCode Web.

Mechanism:
  list_running_servers() → base_url →
  https://kkb-production.jupyter-proxy.kaggle.net/k/<kernel>/<token>/proxy/proxy/<PORT>

Semantics:
  opencode_listening  — TCP 127.0.0.1:PORT accepts connections
  proxy_url_generated — URL string built from Jupyter base_url
  proxy_reachable     — HTTP GET to the public proxy URL got a response
  available           — proxy_reachable (HTTP connectivity only)
  status proxy_http_ok — HTTP probe passed; NOT a claim that OpenCode UI/WebSocket work

JWT/token is ephemeral: may show once to the user; never Dataset/checkpoints/Git.
"""

from __future__ import annotations

import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from .ports import DEFAULT_OPENCODE_PORT, get_opencode_port


def is_port_open(
    host: str,
    port: int,
    timeout: float = 1.0,
) -> bool:
    """Return True if a TCP port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

KAGGLE_JUPYTER_PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"

_JWT_LIKE = re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]*\.[A-Za-z0-9_\-]*")
_TOKEN_SEGMENT = re.compile(r"/k/(\d+)/([^/]+)/")


@dataclass
class AccessInfo:
    available: bool
    url: Optional[str] = None
    url_redacted: Optional[str] = None
    local_port: int = DEFAULT_OPENCODE_PORT
    opencode_listening: bool = False
    proxy_url_generated: bool = False
    proxy_reachable: bool = False
    http_status: Optional[int] = None
    authentication: str = "jupyter_session"
    provider: str = "kaggle_jupyter_proxy"
    status: str = "unavailable"
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self, *, include_url: bool = True) -> dict:
        d = {
            "available": self.available,
            "url_redacted": self.url_redacted or redact_proxy_url(self.url),
            "local_port": self.local_port,
            "opencode_listening": self.opencode_listening,
            "proxy_url_generated": self.proxy_url_generated,
            "proxy_reachable": self.proxy_reachable,
            "http_status": self.http_status,
            "authentication": self.authentication,
            "provider": self.provider,
            "status": self.status,
            "message": self.message,
        }
        if include_url:
            d["url"] = self.url
        return d


def redact_proxy_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    redacted = _JWT_LIKE.sub("<REDACTED_JWT>", url)
    redacted = _TOKEN_SEGMENT.sub(r"/k/\1/<REDACTED>/", redacted)
    return redacted


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def list_jupyter_servers() -> list[dict[str, Any]]:
    try:
        from jupyter_server.serverapp import list_running_servers  # type: ignore

        return list(list_running_servers())
    except Exception:
        try:
            from notebook.notebookapp import list_running_servers  # type: ignore

            return list(list_running_servers())
        except Exception:
            return []


def parse_kernel_token_from_base_url(base_url: str) -> tuple[Optional[str], Optional[str]]:
    if not base_url:
        return None, None
    parts = [p for p in base_url.strip("/").split("/") if p]
    if len(parts) >= 3 and parts[0] == "k":
        return parts[1], parts[2]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


def build_kaggle_proxy_url(
    *,
    kernel: str,
    token: str,
    port: int,
    proxy_host: str = KAGGLE_JUPYTER_PROXY_HOST,
) -> str:
    host = proxy_host.rstrip("/")
    return f"{host}/k/{kernel}/{token}/proxy/proxy/{int(port)}"


def probe_proxy_http(
    url: str,
    *,
    timeout: float = 15.0,
    user_agent: str = "opencode-continuum/5.0-proxy-probe",
) -> tuple[bool, Optional[int], str]:
    req = urllib.request.Request(
        url, method="GET", headers={"User-Agent": user_agent, "Accept": "*/*"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            try:
                resp.read(4096)
            except Exception:
                pass
            if status is not None and int(status) < 500:
                return True, int(status), f"proxy HTTP {status}"
            return False, int(status) if status else None, f"proxy HTTP error {status}"
    except urllib.error.HTTPError as e:
        code = e.code
        if code < 500:
            return True, code, f"proxy HTTP {code}"
        return False, code, f"proxy HTTP error {code}"
    except urllib.error.URLError as e:
        reason = type(e.reason).__name__ if getattr(e, "reason", None) else type(e).__name__
        return False, None, f"proxy unreachable: {reason}"
    except Exception as e:
        return False, None, f"proxy probe failed: {type(e).__name__}"


class KaggleProxyAccess:
    def __init__(
        self,
        port: Optional[int] = None,
        *,
        proxy_host: str = KAGGLE_JUPYTER_PROXY_HOST,
        servers_fn=None,
        probe_fn=None,
        skip_http_probe: bool = False,
    ):
        self.port = int(port if port is not None else get_opencode_port())
        self.proxy_host = proxy_host
        self._servers_fn = servers_fn or list_jupyter_servers
        self._probe_fn = probe_fn or probe_proxy_http
        self.skip_http_probe = skip_http_probe

    def resolve(self) -> AccessInfo:
        listening = wait_for_port("127.0.0.1", self.port, timeout=5.0)
        if not listening:
            return AccessInfo(
                available=False,
                status="opencode_not_listening",
                message=f"Nothing listening on 127.0.0.1:{self.port}",
                local_port=self.port,
                opencode_listening=False,
            )

        servers = self._servers_fn()
        if not servers:
            return AccessInfo(
                available=False,
                status="jupyter_server_not_found",
                message="No Jupyter server from list_running_servers(); cannot build proxy URL",
                local_port=self.port,
                opencode_listening=True,
            )

        base_url = str(servers[0].get("base_url") or "")
        kernel, token = parse_kernel_token_from_base_url(base_url)
        if not kernel or not token:
            return AccessInfo(
                available=False,
                status="proxy_url_generation_failed",
                message="Could not parse kernel/token from Jupyter base_url",
                local_port=self.port,
                opencode_listening=True,
            )

        url = build_kaggle_proxy_url(
            kernel=kernel, token=token, port=self.port, proxy_host=self.proxy_host
        )
        redacted = redact_proxy_url(url)

        if self.skip_http_probe:
            return AccessInfo(
                available=False,
                url=url,
                url_redacted=redacted,
                local_port=self.port,
                opencode_listening=True,
                proxy_url_generated=True,
                proxy_reachable=False,
                status="proxy_url_generated_not_probed",
                message="Proxy URL generated but HTTP probe was skipped (not accessible)",
            )

        reachable, http_status, reason = self._probe_fn(url)
        if not reachable:
            return AccessInfo(
                available=False,
                url=url,
                url_redacted=redacted,
                local_port=self.port,
                opencode_listening=True,
                proxy_url_generated=True,
                proxy_reachable=False,
                http_status=http_status,
                status="proxy_unreachable" if http_status is None else "proxy_http_error",
                message=reason,
            )

        return AccessInfo(
            available=True,
            url=url,
            url_redacted=redacted,
            local_port=self.port,
            opencode_listening=True,
            proxy_url_generated=True,
            proxy_reachable=True,
            http_status=http_status,
            status="proxy_http_ok",
            message=(
                "Public proxy URL answered HTTP (connectivity check). "
                "OpenCode Web UI, WebSocket and streaming are NOT verified by this probe; "
                "validate those during the real Kaggle/browser test. "
                "URL is session-scoped — do not persist it."
            ),
        )


def format_workstation_banner(*, recovery: str, opencode_status: str, access: AccessInfo) -> str:
    lines = [
        "========================================",
        "OpenCode Workstation",
        "========================================",
        f"Workspace: {recovery}",
        f"OpenCode: {opencode_status}",
        f"Access provider: {access.provider}",
    ]
    is_cf = bool(access.provider and access.provider.startswith("cloudflare"))
    if access.available and access.url:
        if is_cf:
            lines.append("OpenCode Web: CLOUDFLARE_READY")
            lines.append("Abrí esta URL en el teléfono (túnel de sesión):")
            lines.append(access.url)
            lines.append("Importante: URL efímera del runtime actual; no la guardes.")
        elif access.proxy_reachable:
            lines.append("OpenCode Web: PROXY_HTTP_OK")
            lines.append("Proxy HTTP respondió. Abrí esta URL desde tu teléfono:")
            lines.append(access.url)
            lines.append("Importante: URL del runtime actual (token de sesión); no la guardes.")
            lines.append("Pendiente de validación real: UI OpenCode, WebSocket, streaming.")
        else:
            lines.append("OpenCode Web: URL_GENERATED")
            lines.append(access.url or "")
    elif access.available and is_cf and access.status == "named_tunnel_running":
        lines.append("OpenCode Web: NAMED_TUNNEL_RUNNING")
        lines.append("Hostname configurado externamente en Cloudflare.")
    elif access.opencode_listening and access.proxy_url_generated and not access.proxy_reachable:
        lines.append("OpenCode Web: NOT_ACCESSIBLE")
        lines.append("OpenCode corre en el runtime; la URL del proxy se generó")
        lines.append("pero el HTTP al proxy falló o no se pudo verificar.")
        lines.append(f"Motivo: {access.status} — {access.message}")
    elif access.opencode_listening:
        lines.append("OpenCode Web: NOT_ACCESSIBLE")
        lines.append("OpenCode corre localmente; no se pudo generar/validar acceso externo.")
        lines.append(f"Motivo: {access.status}")
    else:
        lines.append("OpenCode: NOT_RUNNING")
        lines.append(f"Motivo: {access.status}")
    lines.append("========================================")
    return "\n".join(lines)
