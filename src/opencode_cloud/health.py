"""Real UI health check for OpenCode Web.

TCP open + proxy HTTP OK no garantizan que la SPA cargue (limitación
admitida en access.py). Este módulo hace GET / local y, si hay Chrome,
--headless --dump-dom como en soapp.sh / selenium notebook, pero con
Selenium 4 / Chrome moderno (no selenium==3.141.0 ni chromedriver
LATEST_RELEASE manual).
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional


def verify_local_ui(
    port: int,
    *,
    timeout: float = 15.0,
    http_get=None,
) -> dict:
    """GET http://127.0.0.1:port/ y reporta si parece HTML de la app."""
    url = f"http://127.0.0.1:{int(port)}/"
    try:
        if http_get is not None:
            status, body = http_get(url, timeout=timeout)
        else:
            req = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": "opencode-continuum/5.0-ui-probe"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                body = resp.read(65536)
    except Exception as exc:
        return {
            "ok": False,
            "status": "ui_unreachable",
            "http_status": None,
            "bytes": 0,
            "message": f"GET / falló: {type(exc).__name__}",
            "url": url,
        }
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
    looks_html = "<html" in text.lower() or "<!doctype" in text.lower()
    # OpenCode Web es SPA: el shell HTML suele mencionar opencode o traer <script.
    looks_app = looks_html and ("opencode" in text.lower() or "<script" in text.lower())
    if status is not None and int(status) < 500 and (looks_app or looks_html):
        return {
            "ok": True,
            "status": "ui_html_ok",
            "http_status": int(status),
            "bytes": len(body),
            "message": "GET / devolvió HTML de la app (no verifica WebSocket/SSE).",
            "url": url,
        }
    return {
        "ok": False,
        "status": "ui_unexpected_body",
        "http_status": int(status) if status is not None else None,
        "bytes": len(body),
        "message": "GET / respondió pero el cuerpo no parece la UI (verificá logs).",
        "url": url,
    }


def find_chrome() -> Optional[str]:
    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "/opt/google/chrome/chrome",
    ):
        if name.startswith("/"):
            found = name if Path(name).is_file() else None
        else:
            found = shutil.which(name)
        if found:
            return found
    return None


def chrome_dump_dom(
    url: str,
    *,
    chrome_bin: Optional[str] = None,
    timeout: int = 30,
) -> dict:
    """Chrome headless --dump-dom (patrón soapp.sh --headless).

    Args explícitos, sin invocar shell. Retorna skipped si no hay Chrome.
    """
    binary = chrome_bin or find_chrome()
    if not binary:
        return {
            "ok": False,
            "status": "chrome_missing",
            "message": "Chrome no instalado; solo TCP+HTTP verificado.",
        }
    cmd = [
        binary,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--dump-dom",
        url,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except Exception as exc:
        return {
            "ok": False,
            "status": "chrome_failed",
            "message": f"Chrome falló: {type(exc).__name__}",
        }
    out = proc.stdout or ""
    if proc.returncode == 0 and len(out) > 200:
        return {
            "ok": True,
            "status": "chrome_dom_ok",
            "bytes": len(out),
            "message": "Chrome renderizó DOM (WebSocket/SSE igual requieren test en navegador).",
        }
    return {
        "ok": False,
        "status": "chrome_bad_output",
        "message": (proc.stderr or "")[-500:] or f"exit={proc.returncode}",
    }
