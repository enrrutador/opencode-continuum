"""Permanent web entry: session picker (list, choose, delete) + auto-redirect.

Problem this module solves:
  - OpenCode's project root creates a NEW session on every visit, so
    bookmarking the root URL multiplies sessions (the "chaos").
  - Quick tunnel URLs change per run; even the printed single link churns.

Two cooperating pieces:

1. EntryRedirector: a tiny HTTP server (default port 4097).
   GET / (or /go) -> HTML picker: the list of sessions of the workspace,
   newest first, each linking straight to that session in the SPA, plus a
   delete button per row (DELETE /session/{id}, proxied to the local
   OpenCode API). It never creates sessions. Session links are absolute
   to the app origin when OPENCODE_APP_URL is set (quick tunnel: two
   origins) and relative otherwise (named tunnel: single origin,
   catch-all rule). GET /latest (or /go?auto=1) keeps the previous
   behavior: 302 to the session you left (pin-aware).

2. Session pin: pick_session() prefers the pinned session id persisted
   in the Dataset-backed metadata dir (survives restarts), falling back
   to the most recently updated session. The picker highlights the same
   session with a badge and clears the pin if that session is deleted.

Only the standard library is used.
"""

from __future__ import annotations

import base64
import html
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

DEFAULT_ENTRY_PORT = 4097

HEALTH_PATH = "/healthz"
NO_SESSIONS_PATH = "/session"
LATEST_PATH = "/latest"
PICKER_PATHS = ("/", "/go")
DELETE_PREFIX = "/session/"

_MAX_SESSIONS_IN_PAGE = 100
_TITLE_MAX_CHARS = 140

_MONTHS = (
    "ene", "feb", "mar", "abr", "may", "jun",
    "jul", "ago", "sep", "oct", "nov", "dic",
)


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


def clear_pin(path: Path) -> None:
    """Remove the pin (e.g. the pinned session was just deleted)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def load_pin(path: Path) -> Optional[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sid = data.get("session_id")
    return sid if isinstance(sid, str) and sid else None


def _auth_header() -> Optional[str]:
    password = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
    username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    if not password:
        return None
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def default_fetch_sessions(opencode_port: int, workspace: str) -> list:
    """GET /session?directory=... against the local OpenCode API."""
    url = (
        f"http://127.0.0.1:{opencode_port}/session"
        f"?directory={urllib.parse.quote(str(workspace))}"
    )
    req = urllib.request.Request(url, method="GET")
    auth = _auth_header()
    if auth:
        req.add_header("Authorization", auth)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode() or "null")
    if not isinstance(data, list):
        return []
    return data


def default_delete_session(opencode_port: int, session_id: str) -> bool:
    """DELETE /session/{id} against the local OpenCode API."""
    url = (
        f"http://127.0.0.1:{opencode_port}/session/"
        f"{urllib.parse.quote(session_id, safe='')}"
    )
    req = urllib.request.Request(url, method="DELETE")
    auth = _auth_header()
    if auth:
        req.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def format_relative_time(updated_ms: float, now: Optional[float] = None) -> str:
    """Human-relative Spanish time for a millisecond epoch (UTC dates)."""
    now = now if now is not None else time.time()
    delta = max(0.0, now - float(updated_ms or 0) / 1000.0)
    if delta < 45:
        return "ahora"
    if delta < 3600:
        return f"hace {int(delta // 60)} min"
    if delta < 86400:
        return f"hace {int(delta // 3600)} h"
    if delta < 7 * 86400:
        return f"hace {int(delta // 86400)} d"
    dt = datetime.fromtimestamp(float(updated_ms or 0) / 1000.0, tz=timezone.utc)
    return f"{dt.day} {_MONTHS[dt.month - 1]}"


def _session_row(
    sess: dict,
    ws_b64: str,
    *,
    badge: bool,
    app_base: str,
) -> str:
    sid = str(sess.get("id") or "")
    title = str(sess.get("title") or "").strip()
    if not title:
        title = "Sin título"
    if len(title) > _TITLE_MAX_CHARS:
        title = title[: _TITLE_MAX_CHARS - 1] + "…"
    when = format_relative_time(_session_updated(sess))
    safe_title = html.escape(title, quote=True)
    safe_sid = html.escape(sid, quote=True)
    base = f"{app_base}/{ws_b64}" if app_base else f"/{ws_b64}"
    badge_html = '<span class="badge">última</span>' if badge else ""
    return (
        f'<div class="row-wrap">'
        f'<a class="row" href="{base}/session/{safe_sid}">'
        f'<span class="row-title">{badge_html}{safe_title}</span>'
        f'<span class="row-time">{when}</span>'
        f"</a>"
        f'<button class="del" type="button" data-sid="{safe_sid}" '
        f'data-title="{safe_title}" aria-label="Eliminar sesión">'
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'
        f"</svg></button>"
        f"</div>"
    )


_PICKER_JS = """<script>
document.addEventListener("click", function (e) {
  var btn = e.target.closest(".del");
  if (!btn) return;
  e.preventDefault();
  var sid = btn.getAttribute("data-sid");
  var title = btn.getAttribute("data-title") || "esta sesión";
  if (!confirm('Eliminar "' + title + '"?\\nNo se puede deshacer.')) return;
  btn.disabled = true;
  fetch("/session/" + encodeURIComponent(sid), { method: "DELETE" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); location.reload(); })
    .catch(function () {
      btn.disabled = false;
      alert("No se pudo eliminar la sesión; reintentá en unos segundos.");
    });
});
</script>"""


def render_picker_page(
    sessions: list,
    ws_b64: str,
    pinned_id: Optional[str] = None,
    *,
    app_base: str = "",
) -> str:
    """Self-contained HTML picker: session list, newest first, GET-only links."""
    ordered = sorted(sessions, key=_session_updated, reverse=True)
    chosen = pick_session(sessions, pinned_id)
    chosen_id = (chosen or {}).get("id")
    ordered = [s for s in ordered if s is not chosen]
    if chosen is not None:
        ordered.insert(0, chosen)
    ordered = ordered[:_MAX_SESSIONS_IN_PAGE]

    if ordered:
        rows = "\n".join(
            _session_row(
                s, ws_b64,
                badge=(s.get("id") == chosen_id and bool(chosen_id)),
                app_base=app_base,
            )
            for s in ordered
        )
        empty_html = ""
    else:
        rows = ""
        empty_html = (
            '<p class="empty">Todavía no hay sesiones.<br>'
            "Creá la primera y volvé acá para elegir entre todas.</p>"
        )

    new_href = (
        f"{app_base}/{ws_b64}{NO_SESSIONS_PATH}"
        if app_base
        else f"/{ws_b64}{NO_SESSIONS_PATH}"
    )
    count = len(sessions)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sesiones — OpenCode</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #0b0b0c; color: #ececf1; min-height: 100vh;
}}
.wrap {{ max-width: 680px; margin: 0 auto; padding: 16px 14px 40px; }}
header {{ display: flex; align-items: baseline; gap: 10px; padding: 6px 2px 14px; }}
h1 {{ font-size: 20px; font-weight: 650; letter-spacing: -0.3px; }}
.count {{ color: #8a8a93; font-size: 13px; }}
.badge {{
  background: #ff5c00; color: #0b0b0c; font-size: 11px; font-weight: 700;
  border-radius: 999px; padding: 2px 8px; margin-right: 8px;
  vertical-align: 2px; white-space: nowrap;
}}
.row-wrap {{ display: flex; align-items: stretch; margin-bottom: 8px; }}
.row {{
  display: flex; align-items: center; gap: 12px; text-decoration: none;
  background: #17171a; border: 1px solid #232327; border-radius: 12px;
  padding: 14px 16px; flex: 1; min-width: 0; min-height: 52px;
  transition: background 0.12s ease;
}}
.row:hover, .row:active {{ background: #1f1f24; }}
.row-title {{
  flex: 1; color: #ececf1; font-size: 15px; line-height: 1.35;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden; overflow-wrap: anywhere;
}}
.row-time {{ color: #8a8a93; font-size: 12px; white-space: nowrap; }}
.del {{
  flex: none; width: 46px; margin-left: 8px; border: 1px solid #232327;
  border-radius: 12px; background: #17171a; color: #6f6f78;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: color 0.12s ease, border-color 0.12s ease;
}}
.del svg {{ width: 16px; height: 16px; }}
.del:hover {{ color: #ff4d4f; border-color: #4a2325; }}
.del:disabled {{ opacity: 0.4; }}
.empty {{
  color: #8a8a93; font-size: 15px; line-height: 1.6;
  padding: 28px 4px; text-align: center;
}}
.new {{
  display: block; text-align: center; text-decoration: none; margin-top: 10px;
  color: #ff5c00; font-size: 15px; font-weight: 600;
  border: 1px dashed #4a3527; border-radius: 12px; padding: 13px;
}}
.new:hover, .new:active {{ background: #17130f; }}
footer {{ color: #55555c; font-size: 11px; text-align: center; padding-top: 18px; }}
</style>
</head>
<body>
<div class="wrap">
<header><h1>Sesiones</h1><span class="count">{count} en este workspace</span></header>
{empty_html}{rows}
<a class="new" href="{new_href}">+ Nueva sesión</a>
<footer>OpenCode Continuum — lista en vivo, no crea sesiones</footer>
</div>
{_PICKER_JS}
</body>
</html>"""


class _RedirectorServer(ThreadingHTTPServer):
    """HTTP server carrying its config for the handler."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, config: dict):
        super().__init__(addr, handler)
        self.config = config


class _Handler(BaseHTTPRequestHandler):
    """Picker/redirect logic; never proxies the SPA, never creates sessions."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default stderr noise
        pass

    def _send(
        self,
        code: int,
        location: Optional[str] = None,
        body: str = "",
        content_type: str = "text/plain; charset=utf-8",
        no_store: bool = False,
    ) -> None:
        self.send_response(code)
        if location:
            self.send_header("Location", location)
        payload = b"" if getattr(self, "_head_mode", False) else body.encode()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_HEAD(self):  # noqa: N802 (http.server API)
        self._head_mode = True
        try:
            self.do_GET()
        finally:
            self._head_mode = False

    def _auto_mode(self) -> bool:
        query = (self.path or "").split("?", 1)
        return len(query) > 1 and "auto=1" in query[1]

    def do_GET(self):  # noqa: N802 (http.server API)
        cfg = self.server.config  # type: ignore[attr-defined]
        path = (self.path or "/").split("?")[0]
        head = getattr(self, "_head_mode", False)

        if path == HEALTH_PATH:
            self._send(200, body="ok")
            return

        if path == LATEST_PATH or (path in PICKER_PATHS and self._auto_mode()):
            self._serve_latest(cfg, head)
            return

        if path not in PICKER_PATHS:
            self._send(
                404,
                body="" if head else "usá /go (lista) o /latest (última sesión)",
            )
            return

        self._serve_picker(cfg, head)

    def do_DELETE(self):  # noqa: N802 (http.server API)
        cfg = self.server.config  # type: ignore[attr-defined]
        raw_path = (self.path or "/").split("?")[0]
        if not raw_path.startswith(DELETE_PREFIX):
            self._send_json(404, {"ok": False, "error": "ruta_invalida"})
            return
        sid = urllib.parse.unquote(raw_path[len(DELETE_PREFIX):])
        if not sid or "/" in sid:
            self._send_json(404, {"ok": False, "error": "ruta_invalida"})
            return
        deleted = cfg["delete_session"](sid)
        if not deleted:
            self._send_json(502, {"ok": False, "error": "no_se_pudo_borrar"})
            return
        if cfg["read_pin"]() == sid:
            cfg["clear_pin"]()
        self._send_json(200, {"ok": True, "id": sid})

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(
            code,
            body=json.dumps(payload),
            content_type="application/json",
            no_store=True,
        )

    def _serve_latest(self, cfg: dict, head: bool) -> None:
        try:
            sessions = cfg["fetch_sessions"]()
        except Exception:
            self._send(
                503,
                body="" if head else "OpenCode API no disponible; reintentá en unos minutos.",
            )
            return
        chosen = pick_session(sessions, cfg["read_pin"]())
        if chosen is None:
            self._send(302, location=f"/{cfg['ws_b64']}{NO_SESSIONS_PATH}")
            return
        sid = chosen.get("id")
        if not sid:
            self._send(503, body="" if head else "sesión sin id; reintentá.")
            return
        app_base = cfg["read_app_base"]()
        prefix = f"{app_base}/{cfg['ws_b64']}" if app_base else f"/{cfg['ws_b64']}"
        self._send(302, location=f"{prefix}/session/{sid}")

    def _serve_picker(self, cfg: dict, head: bool) -> None:
        try:
            sessions = cfg["fetch_sessions"]()
        except Exception:
            self._send(
                503,
                content_type="text/html; charset=utf-8",
                body=(
                    "<!DOCTYPE html><html lang=es><head><meta charset=utf-8>"
                    "<meta name=viewport content='width=device-width,initial-scale=1'>"
                    "<title>Sesiones</title></head><body style="
                    "'font-family:system-ui;background:#0b0b0c;color:#8a8a93;"
                    "display:grid;place-items:center;height:100vh;margin:0'>"
                    "<p style='padding:0 20px;text-align:center'>"
                    "OpenCode API no disponible.<br>Reintentá en unos minutos.</p>"
                    "</body></html>"
                ),
                no_store=True,
            )
            return
        page = render_picker_page(
            sessions,
            cfg["ws_b64"],
            cfg["read_pin"](),
            app_base=cfg["read_app_base"](),
        )
        self._send(200, content_type="text/html; charset=utf-8", body=page, no_store=True)


def _env_app_base() -> str:
    return os.environ.get("OPENCODE_APP_URL", "").strip().rstrip("/")


class EntryRedirector:
    """Owns the picker/redirect server lifecycle."""

    def __init__(
        self,
        opencode_port: int,
        workspace: str,
        *,
        port: Optional[int] = None,
        fetch_sessions: Optional[Callable[[], list]] = None,
        delete_session: Optional[Callable[[str], bool]] = None,
        pin_path: Optional[Path] = None,
    ):
        self.opencode_port = int(opencode_port)
        self.workspace = str(workspace)
        self._requested_port = int(port) if port is not None else None
        self._fetch = fetch_sessions
        self._delete = delete_session
        self._pin_path = Path(pin_path) if pin_path else None
        self._server: Optional[_RedirectorServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: Optional[int] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _read_pin(self) -> Optional[str]:
        if self._pin_path is None:
            return None
        return load_pin(self._pin_path)

    def _clear_pin(self) -> None:
        if self._pin_path is not None:
            clear_pin(self._pin_path)

    def start(self) -> None:
        if self.running:
            return
        cfg = {
            "ws_b64": workspace_b64(self.workspace),
            "fetch_sessions": self._fetch
            or (lambda: default_fetch_sessions(self.opencode_port, self.workspace)),
            "delete_session": self._delete
            or (lambda sid: default_delete_session(self.opencode_port, sid)),
            "read_pin": self._read_pin,
            "clear_pin": self._clear_pin,
            "read_app_base": _env_app_base,
        }
        if self._requested_port is not None:
            bind_port = self._requested_port
        else:
            bind_port = int(
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
