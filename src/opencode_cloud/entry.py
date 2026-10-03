"""Permanent web entry ("one link forever") + session pinning.

Problem this module solves:
  - OpenCode's project root creates a NEW session on every visit, so
    bookmarking the root URL multiplies sessions (the "chaos").
  - Quick tunnel URLs change per run; even the printed single link churns.

Two cooperating pieces:

1. EntryRedirector: a tiny read-only HTTP server (default port 4097).
   GET anything -> 302 to /{workspace_b64}/session/{latest_session_id},
   resolved LIVE against the local OpenCode API on every hit. It never
   creates sessions (GET only). With a Cloudflare named tunnel plus a
   dashboard path rule (host /go -> :4097, host * -> :4096), the phone
   bookmark https://<host>/go always lands on the session you left,
   across kernel restarts, tunnels and devices. Fallbacks: no sessions
   yet -> redirect to the session list view; API down -> 503 (watchdog
   or guardian will bring it back; retrying is enough).

2. Session pin: pick_session() prefers the pinned session id persisted
   in the Dataset-backed metadata dir (survives restarts), falling back
   to the most recently updated session. The notebook cell uses it to
   print ONE stable link per boot instead of a list.

Only the standard library is used.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

DEFAULT_ENTRY_PORT = 4097

HEALTH_PATH = "/healthz"
NO_SESSIONS_PATH = "/session"


def workspace_b64(workspace: str) -> str:
    """OpenCode-style url-safe base64 of the workspace path (no padding)."""
    return base64.urlsafe_b64encode(str(workspace).encode()).decode().rstrip("=")


def _session_updated(sess: dict) -> float:
    return float((sess.get("time") or {}).get("updated") or 0)


def pick_session(sessions: list, pinned_id: Optional[str] = None) -> Optional[dict]:
    """Return the session dict to open.

    "Tal cual como lo dejaste" = the most recently updated session. The pin
    only breaks ties: after a restore the unify step can flatten/rewrite
    timestamps, in which case the pinned id (persisted in the Dataset)
    identifies where you actually were. A pinned session that is STALER
    than another one loses: you moved on to a newer chat.
    """
    if not sessions:
        return None
    best = max(sessions, key=_session_updated)
    if pinned_id:
        pinned = next((s for s in sessions if s.get("id") == pinned_id), None)
        if pinned is not None:
            best_updated = _session_updated(best)
            pinned_updated = _session_updated(pinned)
            if pinned_updated == best_updated or best_updated == 0:
                return pinned
    return best


def save_pin(path: Path, session_id: str, title: str = "") -> None:
    """Persist the pinned session (metadata dir is Dataset-checkpointed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session_id": session_id, "title": title}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_pin(path: Path) -> Optional[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sid = data.get("session_id")
    return sid if isinstance(sid, str) and sid else None


def default_fetch_sessions(opencode_port: int, workspace: str) -> list:
    """GET /session?directory=... against the local OpenCode API."""
    url = (
        f"http://127.0.0.1:{opencode_port}/session"
        f"?directory={urllib.parse.quote(str(workspace))}"
    )
    req = urllib.request.Request(url, method="GET")
    password = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
    username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    if password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode() or "null")
    if not isinstance(data, list):
        return []
    return data


class _RedirectorServer(ThreadingHTTPServer):
    """HTTP server carrying its config for the handler."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, config: dict):
        super().__init__(addr, handler)
        self.config = config


class _Handler(BaseHTTPRequestHandler):
    """Read-only redirect logic; never proxies, never creates sessions."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default stderr noise
        pass

    def _send(self, code: int, location: Optional[str] = None, body: str = "") -> None:
        self.send_response(code)
        if location:
            self.send_header("Location", location)
        payload = b"" if getattr(self, "_head_mode", False) else body.encode()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_HEAD(self):  # noqa: N802 (http.server API)
        self._head_mode = True
        try:
            self.do_GET()
        finally:
            self._head_mode = False

    def do_GET(self):  # noqa: N802 (http.server API)
        cfg = self.server.config  # type: ignore[attr-defined]
        path = (self.path or "/").split("?")[0]
        head = getattr(self, "_head_mode", False)

        if path == HEALTH_PATH:
            self._send(200, body="ok" if not head else "")
            return

        if path not in ("/", "/go"):
            # Everything else belongs to OpenCode itself (served on the
            # app port via the tunnel's catch-all rule); we only own
            # the /go entry point.
            self._send(404, body="" if head else "usá /go")
            return

        try:
            sessions = cfg["fetch_sessions"]()
        except Exception:
            self._send(
                503,
                body="" if head else "OpenCode API no disponible; reintentá en unos minutos.",
            )
            return

        chosen = pick_session(sessions)
        if chosen is None:
            self._send(302, location=f"/{cfg['ws_b64']}{NO_SESSIONS_PATH}")
            return

        sid = chosen.get("id")
        if not sid:
            self._send(503, body="" if head else "sesión sin id; reintentá.")
            return
        self._send(302, location=f"/{cfg['ws_b64']}/session/{sid}")


class EntryRedirector:
    """Owns the /go redirect server lifecycle."""

    def __init__(
        self,
        opencode_port: int,
        workspace: str,
        *,
        port: Optional[int] = None,
        fetch_sessions: Optional[Callable[[], list]] = None,
    ):
        self.opencode_port = int(opencode_port)
        self.workspace = str(workspace)
        self._requested_port = int(port) if port is not None else None
        self._fetch = fetch_sessions
        self._server: Optional[_RedirectorServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: Optional[int] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        cfg = {
            "ws_b64": workspace_b64(self.workspace),
            "fetch_sessions": self._fetch
            or (lambda: default_fetch_sessions(self.opencode_port, self.workspace)),
        }
        bind_port = self._requested_port or int(
            os.environ.get("OPENCODE_REDIRECTOR_PORT", str(DEFAULT_ENTRY_PORT))
        )
        self._server = _RedirectorServer(("127.0.0.1", bind_port), _Handler, cfg)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="entry-redirector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=5)
