"""Tests for the external kernel guardian (pure logic, no network/CLI)."""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

GUARDIAN_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "guardian" / "guardian.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("guardian", GUARDIAN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["guardian"] = mod
    spec.loader.exec_module(mod)
    return mod


g = _load()
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    kw.setdefault("kernel", "owner/slug")
    kw.setdefault("push_dir", "/tmp/x")
    return g.GuardianConfig(**kw)


def test_classify_alive():
    out = "ref title author lastRunTime status\nowner/slug t a 2026-10-02 11:00 running"
    assert g.classify_status(out) == "alive"


def test_classify_queued_is_alive():
    assert g.classify_status("status: queued") == "alive"


def test_classify_dead_variants():
    for word in ("complete", "error", "failed", "cancelled"):
        assert g.classify_status(f"status {word}") == "dead", word


def test_classify_unknown():
    assert g.classify_status("some unexpected blob") == "unknown"


def test_extract_last_run_time():
    out = "lastRunTime 2026-10-02 11:00 status complete"
    dt = g.extract_last_run_time(out)
    assert dt == datetime(2026, 10, 2, 11, 0, tzinfo=timezone.utc)


def test_extract_last_run_time_missing():
    assert g.extract_last_run_time("no date here") is None


def test_decide_alive_ok():
    assert g.decide("alive", None, NOW, _cfg())[0] == "ok"


def test_decide_unknown_hold():
    assert g.decide("unknown", None, NOW, _cfg())[0] == "hold"


def test_decide_dead_old_push():
    old = NOW - timedelta(hours=5)
    assert g.decide("dead", old, NOW, _cfg())[0] == "push"


def test_decide_dead_recent_wait():
    recent = NOW - timedelta(minutes=10)
    assert g.decide("dead", recent, NOW, _cfg())[0] == "wait"


def test_decide_dead_no_time_hold_by_default():
    assert g.decide("dead", None, NOW, _cfg())[0] == "hold"


def test_decide_dead_no_time_push_when_allowed():
    action, _ = g.decide("dead", None, NOW, _cfg(allow_unknown_time=True))
    assert action == "push"


def test_decide_future_timestamp_hold():
    future = NOW + timedelta(hours=1)
    assert g.decide("dead", future, NOW, _cfg())[0] == "hold"


def test_looks_like_auth_error():
    for text in ("401 unauthorized", "invalid credentials", "no api key"):
        assert g.looks_like_auth_error(text) is True, text
    assert g.looks_like_auth_error("404 not found") is False


def test_main_disabled_without_kernel(monkeypatch, capsys):
    monkeypatch.delenv("KAGGLE_KERNEL", raising=False)
    assert g.main() == 0
    assert "desactivado" in capsys.readouterr().out


def test_main_disabled_on_auth_error(monkeypatch, capsys):
    import subprocess

    monkeypatch.setenv("KAGGLE_KERNEL", "owner/slug")
    monkeypatch.delenv("GUARDIAN_URL", raising=False)

    def fake_run(argv, timeout=120.0):
        return subprocess.CompletedProcess(argv, 127, "", "invalid credentials")

    monkeypatch.setattr(g, "run_cli", fake_run)
    assert g.main() == 0
    assert "desactivado" in capsys.readouterr().out
