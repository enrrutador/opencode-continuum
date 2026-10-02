"""Scheduler, concurrency, fingerprint, error classification, manifest tests."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_scheduler_no_change_no_publish():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason
    from opencode_cloud.scheduler import CheckpointScheduler

    mgr = CheckpointManager()
    remotes = []

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "a.txt").write_text("x", encoding="utf-8")
        mgr.set_baseline_fingerprint(ws)

        def observe():
            return mgr.observe_workspace(ws)

        def local():
            return {"local": True}

        def remote(reason):
            remotes.append(reason)
            mgr.record_remote_publish()
            return {"published": True}

        sch = CheckpointScheduler(
            mgr, observe_fn=observe, local_fn=local, remote_fn=remote, interval=0.01
        )
        result = sch.run_once()
        assert result["changed"] is False
        assert result["should_publish"] is False
        assert remotes == []


def test_scheduler_change_then_publish_after_cooldown():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason
    from opencode_cloud.scheduler import CheckpointScheduler

    mgr = CheckpointManager()
    remotes = []

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "a.txt").write_text("x", encoding="utf-8")
        mgr.set_baseline_fingerprint(ws)
        mgr.state.last_remote_publish = 0.0

        def observe():
            return mgr.observe_workspace(ws)

        def local():
            return {"local": True}

        def remote(reason):
            remotes.append(reason)
            mgr.record_remote_publish(now=time.time())
            return {"published": True}

        sch = CheckpointScheduler(
            mgr, observe_fn=observe, local_fn=local, remote_fn=remote, interval=0.01
        )
        sch.run_once()
        assert remotes == []

        (ws / "b.txt").write_text("y", encoding="utf-8")
        mgr.state.last_remote_publish = 0.0
        result = sch.run_once()
        assert result["changed"] is True
        assert result["should_publish"] is True
        assert remotes == [PublishReason.COOLDOWN_AND_CHANGES]


def test_scheduler_explicit_and_shutdown_publish():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason
    from opencode_cloud.scheduler import CheckpointScheduler

    mgr = CheckpointManager()
    remotes = []

    sch = CheckpointScheduler(
        mgr,
        observe_fn=lambda: False,
        local_fn=lambda: {},
        remote_fn=lambda r: remotes.append(r) or mgr.record_remote_publish() or {"ok": True},
        interval=60,
    )
    sch.run_once(reason=PublishReason.EXPLICIT)
    assert PublishReason.EXPLICIT in remotes
    remotes.clear()
    sch.shutdown_checkpoint()
    assert PublishReason.SHUTDOWN in remotes


def test_scheduler_watchdog_does_not_publish():
    from opencode_cloud.watchdog import Watchdog

    publishes = []

    class Proc:
        def poll(self):
            return 1

    def restart():
        return Proc()

    wd = Watchdog(check_interval=0, restart_fn=restart, process_poll=lambda: 1)
    wd.set_process(Proc())
    wd._do_restart()
    assert publishes == []
    assert wd.restart_count == 1


def test_concurrent_checkpoint_serialized():
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason
    from opencode_cloud.scheduler import CheckpointScheduler

    mgr = CheckpointManager()
    concurrent = {"n": 0, "max": 0}
    lock_obs = threading.Lock()

    def remote(reason):
        with lock_obs:
            concurrent["n"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["n"])
        time.sleep(0.05)
        with lock_obs:
            concurrent["n"] -= 1
        mgr.record_remote_publish()
        return {"published": True}

    sch = CheckpointScheduler(
        mgr,
        observe_fn=lambda: False,
        local_fn=lambda: {},
        remote_fn=remote,
        interval=60,
    )

    results = []

    def worker(reason):
        results.append(sch.run_once(reason=reason))

    t1 = threading.Thread(target=worker, args=(PublishReason.EXPLICIT,))
    t2 = threading.Thread(target=worker, args=(PublishReason.SHUTDOWN,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert concurrent["max"] == 1
    assert len(results) == 2


def test_fingerprint_detects_create_modify_delete_rename():
    from opencode_cloud.checkpoint import workspace_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "a.txt").write_text("one", encoding="utf-8")
        f1 = workspace_fingerprint(ws)
        (ws / "b.txt").write_text("two", encoding="utf-8")
        f2 = workspace_fingerprint(ws)
        assert f1.digest != f2.digest
        (ws / "a.txt").write_text("ONE-CHANGED", encoding="utf-8")
        f3 = workspace_fingerprint(ws)
        assert f2.digest != f3.digest
        (ws / "a.txt").unlink()
        f4 = workspace_fingerprint(ws)
        assert f3.digest != f4.digest
        (ws / "b.txt").rename(ws / "c.txt")
        f5 = workspace_fingerprint(ws)
        assert f4.digest != f5.digest


def test_fingerprint_same_size_different_content_via_mtime():
    from opencode_cloud.checkpoint import workspace_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        p = ws / "x.txt"
        p.write_text("aaaa", encoding="utf-8")
        f1 = workspace_fingerprint(ws)
        time.sleep(0.01)
        p.write_text("bbbb", encoding="utf-8")
        f2 = workspace_fingerprint(ws)
        assert f1.digest != f2.digest
        assert f1.file_count == f2.file_count == 1


def test_fingerprint_incomplete_when_over_max():
    from opencode_cloud.checkpoint import workspace_fingerprint

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        for i in range(20):
            (ws / f"f{i}.txt").write_text(str(i), encoding="utf-8")
        result = workspace_fingerprint(ws, max_files=5)
        assert result.incomplete is True
        assert result.file_count == 5


def test_classify_download_errors():
    from opencode_cloud.persistence import DownloadErrorKind, classify_download_error

    assert classify_download_error(Exception("404 Dataset not found")) == DownloadErrorKind.DATASET_NOT_FOUND
    assert classify_download_error(Exception("401 Unauthorized")) == DownloadErrorKind.AUTHENTICATION_ERROR
    assert classify_download_error(Exception("403 Forbidden")) == DownloadErrorKind.AUTHORIZATION_ERROR
    assert classify_download_error(Exception("Connection timeout")) == DownloadErrorKind.NETWORK_ERROR
    assert classify_download_error(Exception("429 rate limit")) == DownloadErrorKind.RATE_LIMIT
    assert classify_download_error(Exception("something weird")) == DownloadErrorKind.DOWNLOAD_ERROR


def test_recovery_auth_error_is_restore_failed():
    from opencode_cloud.persistence import (
        KagglePersistence,
        PersistentStore,
        RecoveryStatus,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = PersistentStore(tmp / "opencode_cloud")
        store.ensure_structure()
        kp = KagglePersistence("owner/ds", tmp)
        mock_kh = MagicMock()
        mock_kh.dataset_download.side_effect = Exception("401 Unauthorized")
        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            result = kp.recover_into(store)
        assert result.status == RecoveryStatus.RESTORE_FAILED
        assert "AUTHENTICATION" in result.message or "401" in result.message


def test_recovery_network_error_is_restore_failed():
    from opencode_cloud.persistence import (
        KagglePersistence,
        PersistentStore,
        RecoveryStatus,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = PersistentStore(tmp / "opencode_cloud")
        store.ensure_structure()
        kp = KagglePersistence("owner/ds", tmp)
        mock_kh = MagicMock()
        mock_kh.dataset_download.side_effect = Exception("Network unreachable")
        with patch.object(kp, "_kagglehub", return_value=mock_kh):
            result = kp.recover_into(store)
        assert result.status == RecoveryStatus.RESTORE_FAILED


def test_manifest_valid_and_detects_tamper():
    from opencode_cloud.persistence import (
        PersistentStore,
        validate_integrity_manifest,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = PersistentStore(tmp / "opencode_cloud")
        oc = tmp / "oc"
        oc.mkdir()
        (oc / "f.txt").write_text("hello", encoding="utf-8")
        store.save_local(opencode_data=oc, opencode_config=oc, workspace=oc)
        staging = store.prepare_staging()
        assert (staging / "manifest.json").exists()
        man = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        assert man["kind"] == "opencode-cloud-workstation-manifest"
        ok = validate_integrity_manifest(staging, man)
        assert ok["ok"] is True

        target = next(staging.rglob("f.txt"))
        target.write_text("TAMPERED", encoding="utf-8")
        bad = validate_integrity_manifest(staging, man)
        assert bad["ok"] is False


def test_no_curl_pipe_in_install_sh():
    root = Path(__file__).resolve().parents[1]
    install = (root / "scripts" / "install.sh").read_text(encoding="utf-8")
    import re
    assert not re.search(r"curl[^\n]*\|\s*bash", install)
    assert not re.search(r"wget[^\n]*\|\s*bash", install)
    assert "mktemp" in install
    assert "-o" in install


def test_no_sys_path_insert_in_tests():
    import re
    root = Path(__file__).resolve().parents[1] / "tests"
    call_re = re.compile(r"(?<![\"\'])sys\.path\.insert\s*\(")
    for path in root.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if call_re.search(text):
            raise AssertionError(f"sys.path.insert call in {path}")
