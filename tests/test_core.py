"""Significant tests for OpenCode Cloud Workstation architecture.

Runnable outside Kaggle with mocks. Uses pyproject.toml pythonpath=src.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_no_forbidden_notebook_host_apis_in_source():
    root = Path(__file__).resolve().parents[1] / "src"
    import base64

    forbidden = [
        base64.b64decode(s).decode()
        for s in (
            "Z29vZ2xlLmNvbGFi",
            "ZHJpdmUubW91bnQ=",
            "L2NvbnRlbnQv",
            "TXlEcml2ZQ==",
            "aXNfY29sYWI=",
            "ZnJvbSBnb29nbGUuY29sYWI=",
        )
    ]
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path}:{token}")
    assert not offenders, f"Forbidden notebook-host APIs in source: {offenders}"


def test_no_shell_true_in_source():
    root = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"shell\s*=\s*True", text):
            offenders.append(str(path))
    assert not offenders, f"shell=True found in: {offenders}"


def test_no_bash_c_or_pipe_to_shell_in_source():
    """Reject bash -c, sh -c, curl|bash, wget|sh patterns in runtime/install code."""
    root = Path(__file__).resolve().parents[1] / "src"
    patterns = [
        re.compile(r'["\']bash["\']\s*,\s*["\']-c["\']'),
        re.compile(r'["\']sh["\']\s*,\s*["\']-c["\']'),
        re.compile(r'["\']shell["\']\s*:\s*True|shell\s*=\s*True'),
        re.compile(r'curl[^\n"\']*\|\s*(bash|sh)'),
        re.compile(r'wget[^\n"\']*\|\s*(bash|sh)'),
    ]
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            if pat.search(text):
                offenders.append(f"{path}: {pat.pattern}")
    assert not offenders, f"Unsafe shell patterns: {offenders}"


def test_nvidia_api_key_not_in_subprocess_argv():
    from opencode_cloud import nvidia

    captured_cmds: list[list] = []

    def fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd) if not isinstance(cmd, str) else [cmd])
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        m.stderr = ""
        return m

    fake_key = "nvapi-SECRET-TEST-KEY-DO-NOT-LEAK"

    with patch("subprocess.run", side_effect=fake_run):
        with patch("opencode_cloud.nvidia.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"data":[{"id":"model-a"}]}'
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda *a: None
            mock_open.return_value = mock_resp
            models = nvidia.fetch_models(fake_key)

    assert models == ["model-a"]
    for cmd in captured_cmds:
        joined = " ".join(str(c) for c in cmd)
        assert fake_key not in joined

    cfg = nvidia.build_opencode_config(model="model-a")
    dumped = json.dumps(cfg)
    assert fake_key not in dumped
    assert "{env:NVIDIA_API_KEY}" in dumped


def test_persistent_store_never_writes_kaggle_datasets():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "opencode_cloud"
        from opencode_cloud.persistence import PersistentStore

        store = PersistentStore(root)
        store.ensure_structure()
        assert store.never_writes_to_kaggle_datasets()
        assert not str(store.root).startswith("/kaggle/datasets")


def test_persistent_store_local_save_and_restore():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import PersistentStore

        store = PersistentStore(tmp / "opencode_cloud")
        oc_data = tmp / "oc_data"
        oc_cfg = tmp / "oc_cfg"
        ws = tmp / "ws"
        for d in (oc_data, oc_cfg, ws):
            d.mkdir()
            (d / "file.txt").write_text("hello", encoding="utf-8")

        meta = store.save_local(
            opencode_data=oc_data, opencode_config=oc_cfg, workspace=ws
        )
        assert "saved_at" in meta
        assert store.is_valid_workstation()

        out_data = tmp / "out_data"
        out_cfg = tmp / "out_cfg"
        out_ws = tmp / "out_ws"
        ok = store.restore_to(
            opencode_data=out_data, opencode_config=out_cfg, workspace=out_ws
        )
        assert ok
        assert (out_ws / "file.txt").read_text(encoding="utf-8") == "hello"


def test_prepare_staging_contains_marker():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import PersistentStore

        store = PersistentStore(tmp / "opencode_cloud")
        oc = tmp / "oc"
        oc.mkdir()
        (oc / "x").write_text("1", encoding="utf-8")
        store.save_local(opencode_data=oc, opencode_config=oc, workspace=oc)
        staging = store.prepare_staging()
        assert (staging / "workstation.json").exists()
        assert (staging / "workspace").exists()


def test_validate_workstation_rejects_incomplete():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import PersistentStore

        store = PersistentStore(tmp / "opencode_cloud")
        store.root.mkdir(parents=True)
        (store.root / "workstation.json").write_text(
            json.dumps({"kind": "opencode-cloud-workstation", "version": "5.0.0"}),
            encoding="utf-8",
        )
        result = store.validate_workstation()
        assert not result["ok"]
        assert result["status"] in ("incomplete", "invalid")


def test_validate_workstation_rejects_bad_kind():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import PersistentStore

        store = PersistentStore(tmp / "opencode_cloud")
        store.ensure_structure()
        (store.root / "workstation.json").write_text(
            json.dumps({"kind": "something-else", "version": "5.0.0"}),
            encoding="utf-8",
        )
        result = store.validate_workstation()
        assert not result["ok"]
        assert result["status"] == "incompatible"


def test_publish_refuses_invalid_workstation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import KagglePersistence, PersistentStore

        store = PersistentStore(tmp / "opencode_cloud")
        store.root.mkdir(parents=True)
        (store.root / "workstation.json").write_text(
            json.dumps({"kind": "opencode-cloud-workstation", "version": "5.0.0"}),
            encoding="utf-8",
        )
        kp = KagglePersistence("owner/ds", tmp)
        ok, msg = kp.publish_from_store(store)
        assert not ok
        assert "invalid" in msg.lower() or "refusing" in msg.lower()


def test_kaggle_persistence_download_calls_dataset_download():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import KagglePersistence

        kp = KagglePersistence("owner/ds", tmp)
        fake_path = tmp / "downloaded"
        fake_path.mkdir()
        (fake_path / "workstation.json").write_text(
            json.dumps({"kind": "opencode-cloud-workstation", "version": "5.0.0"}),
            encoding="utf-8",
        )
        (fake_path / "workspace").mkdir()

        mock_kh = MagicMock()
        mock_kh.dataset_download.return_value = str(fake_path)

        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            ok, path, msg = kp.download()
        assert ok
        assert path == fake_path
        mock_kh.dataset_download.assert_called()
        args, kwargs = mock_kh.dataset_download.call_args
        assert args[0] == "owner/ds"


def test_kaggle_persistence_upload_calls_dataset_upload():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import KagglePersistence

        kp = KagglePersistence("owner/ds", tmp)
        staging = tmp / "staging"
        staging.mkdir()
        (staging / "workstation.json").write_text("{}", encoding="utf-8")

        mock_kh = MagicMock()
        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            ok, msg = kp.upload(staging, version_notes="test")
        assert ok
        mock_kh.dataset_upload.assert_called_once()
        args, kwargs = mock_kh.dataset_upload.call_args
        assert args[0] == "owner/ds"
        assert str(staging) in args[1]


def test_kaggle_persistence_download_missing_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        from opencode_cloud.persistence import KagglePersistence

        kp = KagglePersistence("owner/missing", Path(tmp))
        mock_kh = MagicMock()
        mock_kh.dataset_download.side_effect = Exception("404 Not Found")

        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            ok, path, msg = kp.download()
        assert not ok
        assert path is None
        assert "does not exist" in msg.lower() or "not found" in msg.lower()


def test_recovery_fresh_when_dataset_missing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import (
            KagglePersistence,
            PersistentStore,
            RecoveryStatus,
        )

        store = PersistentStore(tmp / "opencode_cloud")
        store.ensure_structure()
        kp = KagglePersistence("owner/missing", tmp)
        mock_kh = MagicMock()
        mock_kh.dataset_download.side_effect = Exception("404 dataset not found")

        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            result = kp.recover_into(store)
        assert result.status == RecoveryStatus.FRESH_WORKSTATION


def test_recovery_restored_from_valid_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import (
            KagglePersistence,
            PersistentStore,
            RecoveryStatus,
        )

        store = PersistentStore(tmp / "opencode_cloud")
        store.ensure_structure()

        download = tmp / "dl"
        download.mkdir()
        (download / "workstation.json").write_text(
            json.dumps({"kind": "opencode-cloud-workstation", "version": "5.0.0"}),
            encoding="utf-8",
        )
        (download / "workspace").mkdir()
        (download / "workspace" / "code.py").write_text("print(1)", encoding="utf-8")
        (download / "state").mkdir()

        kp = KagglePersistence("owner/ds", tmp)
        mock_kh = MagicMock()
        mock_kh.dataset_download.return_value = str(download)

        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            result = kp.recover_into(store)
        assert result.status == RecoveryStatus.RESTORED_FROM_DATASET
        assert (store.workspace / "code.py").exists()


def test_recovery_failed_on_corrupt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from opencode_cloud.persistence import (
            KagglePersistence,
            PersistentStore,
            RecoveryStatus,
        )

        store = PersistentStore(tmp / "opencode_cloud")
        download = tmp / "dl"
        download.mkdir()
        (download / "random.txt").write_text("not a workstation", encoding="utf-8")

        kp = KagglePersistence("owner/ds", tmp)
        mock_kh = MagicMock()
        mock_kh.dataset_download.return_value = str(download)

        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            result = kp.recover_into(store)
        assert result.status == RecoveryStatus.RESTORE_FAILED


def test_checkpoint_local_always_allowed():
    from opencode_cloud.checkpoint import CheckpointManager

    mgr = CheckpointManager()
    mgr.record_local_checkpoint()
    assert mgr.state.last_local_checkpoint > 0


def test_checkpoint_no_change_no_publish():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason

    mgr = CheckpointManager()
    should, reason = mgr.should_publish_remote(now=1_000_000.0)
    assert not should
    assert reason == PublishReason.NONE


def test_checkpoint_significant_change_eligible_after_cooldown():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason

    mgr = CheckpointManager()
    mgr.mark_significant_change()
    mgr.state.last_remote_publish = 0.0
    # Cooldown actual 900s (15min): 400 ya no alcanza, usar 1000.
    should, reason = mgr.should_publish_remote(now=1000.0)
    assert should
    assert reason == PublishReason.COOLDOWN_AND_CHANGES


def test_checkpoint_cooldown_blocks_rapid_publish():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason

    mgr = CheckpointManager()
    mgr.record_remote_publish(now=1000.0)
    mgr.mark_significant_change()
    should, reason = mgr.should_publish_remote(now=1010.0)
    assert not should
    assert reason == PublishReason.NONE


def test_checkpoint_explicit_and_shutdown_bypass_cooldown():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason

    mgr = CheckpointManager()
    mgr.record_remote_publish(now=1000.0)
    should, reason = mgr.should_publish_remote(
        reason=PublishReason.EXPLICIT, now=1001.0
    )
    assert should and reason == PublishReason.EXPLICIT
    should, reason = mgr.should_publish_remote(
        reason=PublishReason.SHUTDOWN, now=1001.0
    )
    assert should and reason == PublishReason.SHUTDOWN


def test_observe_workspace_detects_change():
    from opencode_cloud.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        (ws / "a.txt").write_text("one", encoding="utf-8")

        mgr = CheckpointManager()
        assert not mgr.observe_workspace(ws)
        assert not mgr.observe_workspace(ws)

        (ws / "b.txt").write_text("two", encoding="utf-8")
        assert mgr.observe_workspace(ws) is True
        assert mgr.state.pending_significant_changes >= 1


def test_workspace_fingerprint_stable():
    from opencode_cloud.checkpoint import workspace_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "f.txt").write_text("x", encoding="utf-8")
        a = workspace_fingerprint(ws)
        b = workspace_fingerprint(ws)
        assert a == b
        (ws / "f.txt").write_text("yyyyyyyyyy", encoding="utf-8")
        (ws / "g.txt").write_text("new", encoding="utf-8")
        c = workspace_fingerprint(ws)
        assert a != c


def test_watchdog_restarts_on_death():
    from opencode_cloud.watchdog import Watchdog

    calls = []

    class FakeProc:
        def __init__(self, dead=False):
            self._dead = dead

        def poll(self):
            return 1 if self._dead else None

    current = {"proc": FakeProc(dead=True)}

    def restart():
        calls.append("restart")
        p = FakeProc(dead=False)
        current["proc"] = p
        return p

    wd = Watchdog(
        check_interval=0,
        restart_fn=restart,
        process_poll=lambda: current["proc"].poll(),
    )
    wd.set_process(current["proc"])
    wd._do_restart()
    assert calls == ["restart"]
    assert wd.restart_count == 1
    assert wd.last_restart_ok is True


def test_watchdog_requires_real_restart_fn():
    from opencode_cloud.watchdog import Watchdog

    wd = Watchdog(restart_fn=None)
    with pytest.raises(RuntimeError):
        wd.start()


def test_secrets_env_fallback():
    from opencode_cloud.secrets import EnvSecrets

    mgr = EnvSecrets()
    assert mgr.get("NONEXISTENT_XYZ") is None
    os.environ["TEST_SECRET_XYZ"] = "value"
    try:
        assert mgr.get("TEST_SECRET_XYZ") == "value"
    finally:
        del os.environ["TEST_SECRET_XYZ"]


def test_kaggle_secrets_uses_user_secrets_client():
    from opencode_cloud.secrets import KaggleSecrets

    mock_client = MagicMock()
    mock_client.get_secret.return_value = "secret-value"

    with patch.dict("sys.modules", {"kaggle_secrets": MagicMock()}):
        import sys

        sys.modules["kaggle_secrets"].UserSecretsClient = MagicMock(
            return_value=mock_client
        )
        ks = KaggleSecrets.__new__(KaggleSecrets)
        ks._client = mock_client
        assert ks.get("NVIDIA_API_KEY") == "secret-value"
        mock_client.get_secret.assert_called_with("NVIDIA_API_KEY")


def test_github_credential_file_deleted_even_on_failure():
    from opencode_cloud.github_sync import init_repo, temporary_credential_helper

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        init_repo(ws)
        seen = {}

        try:
            with temporary_credential_helper("fake-token", ws) as cred:
                seen["path"] = cred
                assert cred.exists()
                raise RuntimeError("simulate git failure")
        except RuntimeError:
            pass

        assert "path" in seen
        assert not seen["path"].exists()


def test_github_validate_url():
    from opencode_cloud.github_sync import validate_repo_url

    assert validate_repo_url("https://github.com/owner/repo").endswith(".git")
    with pytest.raises(ValueError):
        validate_repo_url("not-a-url")


def test_write_opencode_config_incremental():
    from opencode_cloud.opencode import write_opencode_config

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "opencode.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "provider": {"other": {"name": "Other"}},
                    "permission": "ask",
                }
            ),
            encoding="utf-8",
        )
        write_opencode_config(cfg_path, model="nemotron-test")
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["permission"] == "allow"
        assert "nvidia" in data["provider"]
        assert "other" in data["provider"]
        assert data["model"] == "nvidia/nemotron-test"
        dumped = json.dumps(data)
        assert "nvapi-" not in dumped
        assert "{env:NVIDIA_API_KEY}" in dumped


def test_runtime_paths_kaggle_layout():
    from opencode_cloud.runtime import get_paths

    paths = get_paths("kaggle")
    assert paths.working == Path("/kaggle/working")
    assert paths.cloud_root == Path("/kaggle/working/opencode_cloud")
    assert paths.workspace == Path("/kaggle/working/opencode_cloud/workspace")


def test_ensure_node_idempotent_when_present():
    import shutil

    from opencode_cloud.opencode import ensure_node

    if shutil.which("node"):
        ver = ensure_node()
        assert ver.startswith("v") or ver[0].isdigit()
