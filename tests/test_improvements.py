"""Tests for A1/B1/C2/A3 improvements on top of v13 (no network)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from opencode_cloud import health
from opencode_cloud.persistence import (
    latest_session_id,
    unify_live_dbs,
    unify_opencode_sessions_to_latest_project,
)
from opencode_cloud.vscode_tunnel import enabled_by_env, ensure_vscode_cli


def test_health_html_ok():
    def fake_get(url, timeout=15.0):
        return 200, b"<html><body><script>opencode</script></body></html>"

    res = health.verify_local_ui(4096, http_get=fake_get)
    assert res["ok"] is True
    assert res["status"] == "ui_html_ok"


def test_health_unexpected_body():
    def fake_get(url, timeout=15.0):
        return 200, b"{}"

    res = health.verify_local_ui(4096, http_get=fake_get)
    assert res["ok"] is False


def test_health_unreachable():
    def fake_get(url, timeout=15.0):
        raise OSError("down")

    res = health.verify_local_ui(4096, http_get=fake_get)
    assert res["status"] == "ui_unreachable"


def test_chrome_missing_returns_skipped(monkeypatch):
    monkeypatch.setattr(health, "find_chrome", lambda: None)
    res = health.chrome_dump_dom("http://127.0.0.1:4096/")
    assert res["status"] == "chrome_missing"


def _make_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE project (id TEXT)")
        con.execute("CREATE TABLE session (id TEXT, project_id TEXT)")
        con.execute("INSERT INTO project (id) VALUES ('p-old')")
        con.execute("INSERT INTO project (id) VALUES ('p-new')")
        con.execute("INSERT INTO session (id, project_id) VALUES ('s1', 'p-old')")
        con.execute("INSERT INTO session (id, project_id) VALUES ('s2', 'p-old')")
        con.commit()
    finally:
        con.close()


def test_unify_and_latest_session(tmp_path):
    db = tmp_path / "opencode.db"
    _make_db(db)
    res = unify_opencode_sessions_to_latest_project(db)
    assert res.get("migrated") == 2
    assert latest_session_id(db) in ("s1", "s2")
    out = unify_live_dbs(tmp_path, None)
    assert out.get("migrated") == 0


def test_latest_session_missing_returns_none(tmp_path):
    assert latest_session_id(tmp_path / "nope.db") is None


def test_cloudflared_override_respected(tmp_path, monkeypatch):
    from opencode_cloud import cloudflare_access as cfa

    fake = tmp_path / "cloudflared"
    fake.write_bytes(b"x" * 10)
    fake.chmod(0o755)
    monkeypatch.setenv("OPENCODE_CLOUDFLARED_BIN", str(fake))
    assert cfa.ensure_cloudflared() == str(fake)


def test_vscode_cli_reuses_existing(tmp_path):
    dest = tmp_path / "cli"
    dest.mkdir()
    binary = dest / "code"
    binary.write_text("#!/bin/sh\necho hi\n")
    binary.chmod(0o755)
    assert ensure_vscode_cli(dest) == str(binary)


def test_vscode_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENCODE_VSCODE_TUNNEL", raising=False)
    assert enabled_by_env() is False
