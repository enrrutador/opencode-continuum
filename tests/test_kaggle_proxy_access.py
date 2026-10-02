"""Unit tests for Kaggle Jupyter Proxy access."""

from __future__ import annotations

from opencode_cloud.access import (
    AccessInfo,
    KaggleProxyAccess,
    build_kaggle_proxy_url,
    format_workstation_banner,
    parse_kernel_token_from_base_url,
    redact_proxy_url,
)


def test_parse_kernel_token():
    k, t = parse_kernel_token_from_base_url("/k/12345/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4.abc/")
    assert k == "12345"
    assert t.startswith("eyJ")


def test_build_url():
    url = build_kaggle_proxy_url(kernel="99", token="tok", port=4096)
    assert url.endswith("/k/99/tok/proxy/proxy/4096")
    assert "jupyter-proxy.kaggle.net" in url


def test_redact_hides_token():
    url = build_kaggle_proxy_url(kernel="7", token="SUPERSECRETTOKEN", port=8000)
    r = redact_proxy_url(url)
    assert "SUPERSECRETTOKEN" not in (r or "")
    assert "REDACTED" in (r or "")


def test_available_false_when_only_url_generated_not_probed(monkeypatch):
    from opencode_cloud import access as m

    monkeypatch.setattr(m, "wait_for_port", lambda *a, **k: True)
    info = KaggleProxyAccess(
        4096,
        servers_fn=lambda: [{"base_url": "/k/1/secrettoken/"}],
        skip_http_probe=True,
    ).resolve()
    assert info.proxy_url_generated is True
    assert info.available is False


def test_available_true_only_after_successful_probe(monkeypatch):
    from opencode_cloud import access as m

    monkeypatch.setattr(m, "wait_for_port", lambda *a, **k: True)

    def probe(url, **kw):
        return True, 200, "proxy HTTP 200"

    info = KaggleProxyAccess(
        4096,
        servers_fn=lambda: [{"base_url": "/k/42/tok42/"}],
        probe_fn=probe,
    ).resolve()
    assert info.available is True
    assert info.proxy_reachable is True
    assert info.status == "proxy_http_ok"


def test_proxy_unreachable(monkeypatch):
    from opencode_cloud import access as m

    monkeypatch.setattr(m, "wait_for_port", lambda *a, **k: True)

    def probe(url, **kw):
        return False, None, "proxy unreachable: TimeoutError"

    info = KaggleProxyAccess(
        4096,
        servers_fn=lambda: [{"base_url": "/k/1/t/"}],
        probe_fn=probe,
    ).resolve()
    assert info.available is False
    assert info.status == "proxy_unreachable"


def test_no_jupyter(monkeypatch):
    from opencode_cloud import access as m

    monkeypatch.setattr(m, "wait_for_port", lambda *a, **k: True)
    info = KaggleProxyAccess(4096, servers_fn=lambda: []).resolve()
    assert info.status == "jupyter_server_not_found"


def test_not_listening(monkeypatch):
    from opencode_cloud import access as m

    monkeypatch.setattr(m, "wait_for_port", lambda *a, **k: False)
    info = KaggleProxyAccess(59999, servers_fn=lambda: [{"base_url": "/k/1/t/"}]).resolve()
    assert info.status == "opencode_not_listening"


def test_banner_proxy_http_ok_when_reachable():
    info = AccessInfo(
        available=True,
        url="https://example/k/1/tok/proxy/proxy/4096",
        proxy_reachable=True,
        proxy_url_generated=True,
        opencode_listening=True,
        status="proxy_http_ok",
    )
    text = format_workstation_banner(recovery="FRESH", opencode_status="RUNNING", access=info)
    assert "PROXY_HTTP_OK" in text


def test_access_module_is_kaggle_proxy_only():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "src/opencode_cloud/access.py").read_text()
    assert "KaggleProxyAccess" in text
    assert "jupyter-proxy.kaggle.net" in text or "KAGGLE_JUPYTER_PROXY" in text
    assert "cloudflared" not in text.lower()
    assert "ngrok" not in text.lower()
