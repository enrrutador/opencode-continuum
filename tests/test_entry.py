"""Tests for entry.py: /go redirector + session pin (no network needed)."""

import http.client
import json
import threading
from pathlib import Path

from opencode_cloud.entry import (
    EntryRedirector,
    load_pin,
    pick_session,
    save_pin,
    workspace_b64,
)


def _sess(sid, updated):
    return {"id": sid, "title": sid, "time": {"updated": updated}}


SESSIONS = [_sess("old", 100), _sess("latest", 300), _sess("mid", 200)]


def test_pick_session_latest_by_default():
    assert pick_session(SESSIONS)["id"] == "latest"


def test_pick_session_pinned_wins_when_it_is_latest():
    assert pick_session(SESSIONS, "latest")["id"] == "latest"


def test_pick_session_stale_pinned_loses():
    assert pick_session(SESSIONS, "old")["id"] == "latest"


def test_pick_session_pinned_wins_on_flattened_timestamps():
    flat = [_sess("a", 0), _sess("b", 0), _sess("c", 0)]
    assert pick_session(flat, "b")["id"] == "b"


def test_pick_session_pinned_missing_falls_back():
    assert pick_session(SESSIONS, "gone")["id"] == "latest"


def test_pick_session_empty():
    assert pick_session([], "x") is None
    assert pick_session([]) is None


def test_workspace_b64_matches_opencode_style():
    import base64

    expected = base64.urlsafe_b64encode(b"/kaggle/working/x").decode().rstrip("=")
    assert workspace_b64("/kaggle/working/x") == expected


def test_pin_roundtrip(tmp_path: Path):
    pin = tmp_path / "metadata" / "session.json"
    save_pin(pin, "ses_1", "titulo")
    assert load_pin(pin) == "ses_1"
    save_pin(pin, "ses_2", "otro")
    assert load_pin(pin) == "ses_2"


def test_load_pin_missing_or_corrupt(tmp_path: Path):
    assert load_pin(tmp_path / "no.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{no json", encoding="utf-8")
    assert load_pin(bad) is None


def _get_free_redirector(fetched):
    redirector = EntryRedirector(
        4096, "/kaggle/working/ws", port=0, fetch_sessions=fetched
    )
    redirector.start()
    assert redirector.running and redirector.port
    return redirector


def _request(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    headers = dict(resp.getheaders())
    conn.close()
    return resp.status, headers, body


def test_redirector_latest_session_302():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        status, headers, _ = _request(r.port, "/go")
        ws = workspace_b64("/kaggle/working/ws")
        assert status == 302
        assert headers["Location"] == f"/{ws}/session/latest"
    finally:
        r.stop()
    assert not r.running


def test_redirector_no_sessions_redirects_to_list():
    r = _get_free_redirector(lambda: [])
    try:
        status, headers, _ = _request(r.port, "/")
        assert status == 302
        assert headers["Location"].endswith("/session")
    finally:
        r.stop()


def test_redirector_api_down_503():
    def boom():
        raise RuntimeError("api down")

    r = _get_free_redirector(boom)
    try:
        status, _, body = _request(r.port, "/go")
        assert status == 503
        assert "reintent" in body
    finally:
        r.stop()


def test_redirector_healthz_ok():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        status, _, body = _request(r.port, "/healthz")
        assert status == 200 and body == "ok"
    finally:
        r.stop()


def test_redirector_other_paths_404():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        status, _, body = _request(r.port, "/session/abc")
        assert status == 404
        assert "/go" in body
    finally:
        r.stop()


def test_redirector_never_creates_sessions():
    calls = []

    def fetch():
        calls.append(1)
        return SESSIONS

    r = _get_free_redirector(fetch)
    try:
        _request(r.port, "/go")
        assert len(calls) == 1
    finally:
        r.stop()


def test_redirector_head_support():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", r.port, timeout=5)
        conn.request("HEAD", "/go")
        resp = conn.getresponse()
        status = resp.status
        loc = resp.getheader("Location")
        resp.read()
        conn.close()
        assert status == 302
        assert loc.endswith("/session/latest")
    finally:
        r.stop()


def test_redirector_start_stop_idempotent():
    r = _get_free_redirector(lambda: SESSIONS)
    port = r.port
    r.start()
    assert r.port == port
    r.stop()
    r.stop()
    assert not r.running


def test_threads_are_daemon():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        names = [t.name for t in threading.enumerate()]
        assert "entry-redirector" in names
    finally:
        r.stop()


def test_pin_json_shape(tmp_path: Path):
    pin = tmp_path / "session.json"
    save_pin(pin, "s1", "T")
    data = json.loads(pin.read_text(encoding="utf-8"))
    assert data == {"session_id": "s1", "title": "T"}
