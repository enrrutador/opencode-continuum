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
        status, headers, _ = _request(r.port, "/latest")
        ws = workspace_b64("/kaggle/working/ws")
        assert status == 302
        assert headers["Location"] == f"/{ws}/session/latest"
    finally:
        r.stop()
    assert not r.running


def test_redirector_go_auto_query_still_redirects():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        status, headers, _ = _request(r.port, "/go?auto=1")
        ws = workspace_b64("/kaggle/working/ws")
        assert status == 302
        assert headers["Location"] == f"/{ws}/session/latest"
    finally:
        r.stop()


def test_redirector_no_sessions_latest_redirects_to_list():
    r = _get_free_redirector(lambda: [])
    try:
        status, headers, _ = _request(r.port, "/latest")
        assert status == 302
        assert headers["Location"].endswith("/session")
    finally:
        r.stop()


def test_redirector_api_down_503():
    def boom():
        raise RuntimeError("api down")

    r = _get_free_redirector(boom)
    try:
        status, _, body = _request(r.port, "/latest")
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
        conn.request("HEAD", "/latest")
        resp = conn.getresponse()
        status = resp.status
        loc = resp.getheader("Location")
        resp.read()
        conn.close()
        assert status == 302
        assert loc.endswith("/session/latest")
    finally:
        r.stop()


def _get_picker(port: int):
    status, headers, body = _request(port, "/go")
    return status, headers, body


def test_picker_serves_html_list():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        status, headers, body = _get_picker(r.port)
        ws = workspace_b64("/kaggle/working/ws")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert headers["Cache-Control"] == "no-store"
        assert f'/{ws}/session/latest' in body
        assert f'/{ws}/session/mid' in body
        assert f'/{ws}/session/old' in body
        assert body.index("latest") < body.index("mid") < body.index("old")
        assert "+ Nueva sesión" in body
        assert f'href="/{ws}/session"' in body
    finally:
        r.stop()


def test_picker_badges_latest_session():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        _, _, body = _get_picker(r.port)
        assert "última" in body
        assert body.count('class="badge"') == 1
    finally:
        r.stop()


def test_picker_badge_follows_pin_on_flat_timestamps():
    flat = [_sess("a", 0), _sess("b", 0), _sess("c", 0)]
    pin_file = None
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pin_file = Path(td) / "session.json"
        save_pin(pin_file, "b", "B")
        r = EntryRedirector(
            4096, "/kaggle/working/ws", port=0,
            fetch_sessions=lambda: flat, pin_path=pin_file,
        )
        r.start()
        try:
            _, _, body = _get_picker(r.port)
            assert body.count('class="badge"') == 1
            first_link = body.split('class="row"', 1)[1]
            assert "/session/b" in first_link
        finally:
            r.stop()


def test_picker_escapes_titles():
    evil = [{
        "id": "x1",
        "title": "<script>alert(1)</script>",
        "time": {"updated": 100},
    }]
    r = _get_free_redirector(lambda: evil)
    try:
        _, _, body = _get_picker(r.port)
        assert "<script>alert" not in body
        import html as _html

        assert _html.escape("<script>alert(1)</script>") in body
    finally:
        r.stop()


def test_picker_untitled_sessions_render_fallback():
    sess = [{"id": "n1", "title": "", "time": {"updated": 100}}]
    r = _get_free_redirector(lambda: sess)
    try:
        _, _, body = _get_picker(r.port)
        assert "Sin título" in body
    finally:
        r.stop()


def test_picker_empty_state():
    r = _get_free_redirector(lambda: [])
    try:
        status, _, body = _get_picker(r.port)
        assert status == 200
        assert "Todavía no hay sesiones" in body
        ws = workspace_b64("/kaggle/working/ws")
        assert f'href="/{ws}/session"' in body
    finally:
        r.stop()


def test_picker_api_down_503_html():
    def boom():
        raise RuntimeError("api down")

    r = _get_free_redirector(boom)
    try:
        status, headers, body = _get_picker(r.port)
        assert status == 503
        assert headers["Content-Type"].startswith("text/html")
        assert "Reintentá" in body
    finally:
        r.stop()


def test_picker_head_support():
    r = _get_free_redirector(lambda: SESSIONS)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", r.port, timeout=5)
        conn.request("HEAD", "/go")
        resp = conn.getresponse()
        status = resp.status
        resp.read()
        conn.close()
        assert status == 200
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
