"""Access layer coherence tests (Kaggle proxy only)."""

from __future__ import annotations

import re
from pathlib import Path


def test_no_external_tunnel_modules():
    root = Path(__file__).resolve().parents[1] / "src" / "opencode_cloud"
    assert not (root / "access_cloudflare.py").exists()
    assert not (root / "access_ngrok.py").exists()


def test_no_external_tunnel_imports_in_src():
    root = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "access_cloudflare" in text or "CloudflareAccessLayer" in text:
            offenders.append(str(path))
        if re.search(r"allow_cloudflare_fallback", text):
            offenders.append(str(path))
        if "ngrok" in text.lower() and "test" not in path.name:
            offenders.append(str(path))
    assert not offenders, offenders


def test_primary_is_kaggle_proxy():
    from opencode_cloud.access import KAGGLE_JUPYTER_PROXY_HOST, KaggleProxyAccess

    assert "jupyter-proxy.kaggle.net" in KAGGLE_JUPYTER_PROXY_HOST
    assert callable(KaggleProxyAccess)
