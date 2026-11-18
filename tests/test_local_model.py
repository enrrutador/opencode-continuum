"""Tests for the optional local model integration."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_install_local_model_writes_provider_and_restarts(tmp_path, monkeypatch):
    import opencode_kaggle.local_model as lm

    cfg = tmp_path / "opencode" / "opencode.jsonc"
    monkeypatch.setattr(lm, "ensure_model_server", lambda model, port: "http://127.0.0.1:8000")
    monkeypatch.setattr(lm, "restart_opencode", lambda port: True)
    monkeypatch.setattr(lm, "verify_model", lambda provider, model: True)
    monkeypatch.setattr(
        "opencode_cloud.runtime.get_paths", lambda: SimpleNamespace(xdg_config=tmp_path)
    )
    monkeypatch.setenv("LOCAL_MODEL_ID", "org/9b")

    assert lm.install_local_model() is True

    data = json.loads(cfg.read_text(encoding="utf-8"))
    provider = data["provider"]["kaggle-deepseek"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
    assert provider["models"]["org/9b"]["name"] == "DeepSeek-V4-Pro 9B (heretic)"


def test_install_local_model_skips_without_server(tmp_path, monkeypatch):
    import opencode_kaggle.local_model as lm

    monkeypatch.setattr(lm, "ensure_model_server", lambda model, port: "")
    monkeypatch.setattr(
        "opencode_cloud.runtime.get_paths", lambda: SimpleNamespace(xdg_config=tmp_path)
    )

    assert lm.install_local_model() is False
    assert not (tmp_path / "opencode" / "opencode.jsonc").exists()


def test_port_open_detects_closed_port():
    import opencode_kaggle.local_model as lm

    assert lm.port_open("127.0.0.1", 1, timeout=0.2) is False
