"""Integration-style tests for the full persistence pipeline.

Uses a fake file-backed kagglehub to simulate a real Dataset
without requiring network or a Kaggle account.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class FakeKaggleHub:
    """Simulates kagglehub Dataset storage as a local directory of versions."""

    def __init__(self, remote_root: Path):
        self.remote_root = Path(remote_root)
        self.remote_root.mkdir(parents=True, exist_ok=True)
        self.version = 0
        self.download_calls = []
        self.upload_calls = []
        self.fail_download = False
        self.fail_upload = False
        self.not_found = False

    def dataset_download(self, handle, path=None, force_download=False, output_dir=None):
        self.download_calls.append(
            {"handle": handle, "force": force_download, "output_dir": output_dir}
        )
        if self.not_found:
            raise Exception("404 Dataset not found")
        if self.fail_download:
            raise Exception("network error")
        if self.version == 0:
            raise Exception("404 does not exist")
        vdir = self.remote_root / f"v{self.version}"
        if output_dir:
            dest = Path(output_dir) / f"v{self.version}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(vdir, dest)
            return str(dest)
        return str(vdir)

    def dataset_upload(self, handle, local_dataset_dir, version_notes=""):
        self.upload_calls.append(
            {
                "handle": handle,
                "local": str(local_dataset_dir),
                "notes": version_notes,
            }
        )
        if self.fail_upload:
            raise Exception("upload failed")
        self.version += 1
        vdir = self.remote_root / f"v{self.version}"
        if vdir.exists():
            shutil.rmtree(vdir)
        shutil.copytree(local_dataset_dir, vdir)
        return None


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        working = tmp / "working"
        working.mkdir()
        remote = tmp / "remote_dataset"
        remote.mkdir()
        hub = FakeKaggleHub(remote)
        yield {"tmp": tmp, "working": working, "hub": hub}


def _seed_runtime_files(tmp: Path):
    oc_data = tmp / "home_oc_data"
    oc_cfg = tmp / "home_oc_cfg"
    ws = tmp / "home_ws"
    for d in (oc_data, oc_cfg, ws):
        d.mkdir(parents=True, exist_ok=True)
    (oc_data / "session.json").write_text('{"s":1}', encoding="utf-8")
    (oc_cfg / "opencode.json").write_text(
        json.dumps({"provider": {"nvidia": {"options": {"apiKey": "{env:NVIDIA_API_KEY}"}}}}),
        encoding="utf-8",
    )
    (ws / "main.py").write_text("print('hello')", encoding="utf-8")
    return oc_data, oc_cfg, ws


def test_full_cycle_fresh_then_restore(env):
    from opencode_cloud.persistence import (
        KagglePersistence,
        PersistentStore,
        RecoveryStatus,
    )

    working = env["working"]
    hub = env["hub"]
    tmp = env["tmp"]

    store = PersistentStore(working / "opencode_cloud")
    store.ensure_structure()
    kp = KagglePersistence("owner/ws", working)

    with patch.object(kp, "_kagglehub", return_value=hub):
        result = kp.recover_into(store)
    assert result.status == RecoveryStatus.FRESH_WORKSTATION

    oc_data, oc_cfg, ws = _seed_runtime_files(tmp)
    store.save_local(opencode_data=oc_data, opencode_config=oc_cfg, workspace=ws)
    assert store.is_valid_workstation()

    with patch.object(kp, "_kagglehub", return_value=hub):
        ok, msg = kp.publish_from_store(store, version_notes="first")
    assert ok, msg
    assert hub.version == 1

    working2 = tmp / "working2"
    working2.mkdir()
    store2 = PersistentStore(working2 / "opencode_cloud")
    kp2 = KagglePersistence("owner/ws", working2)
    with patch.object(kp2, "_kagglehub", return_value=hub):
        result2 = kp2.recover_into(store2)
    assert result2.status == RecoveryStatus.RESTORED_FROM_DATASET
    assert (store2.workspace / "main.py").read_text(encoding="utf-8") == "print('hello')"


def test_corrupt_dataset_does_not_pretend_success(env):
    from opencode_cloud.persistence import (
        KagglePersistence,
        PersistentStore,
        RecoveryStatus,
    )

    working = env["working"]
    hub = env["hub"]
    hub.version = 1
    vdir = hub.remote_root / "v1"
    vdir.mkdir()
    (vdir / "junk.txt").write_text("nope", encoding="utf-8")

    store = PersistentStore(working / "opencode_cloud")
    kp = KagglePersistence("owner/ws", working)
    with patch.object(kp, "_kagglehub", return_value=hub):
        result = kp.recover_into(store)
    assert result.status == RecoveryStatus.RESTORE_FAILED


def test_checkpoint_manager_gates_remote_publish(env):
    from opencode_cloud.checkpoint import CheckpointManager, PublishReason

    mgr = CheckpointManager()
    mgr.record_remote_publish(now=0.0)
    mgr.mark_significant_change()
    should, reason = mgr.should_publish_remote(now=10.0)
    assert not should
    should, reason = mgr.should_publish_remote(now=1000.0)
    assert should
    assert reason == PublishReason.COOLDOWN_AND_CHANGES


def test_upload_argument_order_handle_then_path(env):
    from opencode_cloud.persistence import KagglePersistence, PersistentStore

    working = env["working"]
    hub = env["hub"]
    tmp = env["tmp"]
    store = PersistentStore(working / "opencode_cloud")
    oc_data, oc_cfg, ws = _seed_runtime_files(tmp)
    store.save_local(opencode_data=oc_data, opencode_config=oc_cfg, workspace=ws)
    kp = KagglePersistence("owner/ws", working)
    with patch.object(kp, "_kagglehub", return_value=hub):
        ok, _ = kp.publish_from_store(store)
    assert ok
    assert hub.upload_calls
    call = hub.upload_calls[0]
    assert call["handle"] == "owner/ws"


def test_second_runtime_continues_workspace(env):
    from opencode_cloud.persistence import (
        KagglePersistence,
        PersistentStore,
        RecoveryStatus,
    )

    working = env["working"]
    hub = env["hub"]
    tmp = env["tmp"]

    store_a = PersistentStore(working / "opencode_cloud")
    kp = KagglePersistence("owner/ws", working)
    oc_data, oc_cfg, ws = _seed_runtime_files(tmp)
    (ws / "project.md").write_text("# Project A", encoding="utf-8")
    store_a.save_local(opencode_data=oc_data, opencode_config=oc_cfg, workspace=ws)
    with patch.object(kp, "_kagglehub", return_value=hub):
        ok, _ = kp.publish_from_store(store_a, version_notes="runtime-a")
    assert ok

    working_b = tmp / "working_b"
    working_b.mkdir()
    store_b = PersistentStore(working_b / "opencode_cloud")
    kp_b = KagglePersistence("owner/ws", working_b)
    with patch.object(kp_b, "_kagglehub", return_value=hub):
        result = kp_b.recover_into(store_b)
    assert result.status == RecoveryStatus.RESTORED_FROM_DATASET
    assert (store_b.workspace / "project.md").read_text(encoding="utf-8") == "# Project A"


def test_watchdog_does_not_publish_dataset(env):
    from opencode_cloud.watchdog import Watchdog

    hub = env["hub"]
    restarts = []

    class Proc:
        def __init__(self, dead=False):
            self._dead = dead

        def poll(self):
            return 1 if self._dead else None

    current = {"p": Proc(dead=True)}

    def restart():
        restarts.append(1)
        current["p"] = Proc(dead=False)
        return current["p"]

    wd = Watchdog(
        check_interval=0,
        restart_fn=restart,
        process_poll=lambda: current["p"].poll(),
    )
    wd.set_process(current["p"])
    wd._do_restart()
    assert restarts == [1]
    assert hub.upload_calls == []


def test_secrets_never_land_in_store(env):
    from opencode_cloud.opencode import write_opencode_config
    from opencode_cloud.persistence import PersistentStore

    working = env["working"]
    tmp = env["tmp"]
    store = PersistentStore(working / "opencode_cloud")

    oc_data = tmp / "oc_data"
    oc_cfg = tmp / "oc_cfg"
    ws = tmp / "ws"
    oc_data.mkdir()
    oc_cfg.mkdir()
    ws.mkdir()
    write_opencode_config(oc_cfg / "opencode.json", model="nemotron-x")
    (ws / "code.py").write_text("print('hi')", encoding="utf-8")

    store.save_local(opencode_data=oc_data, opencode_config=oc_cfg, workspace=ws)

    for path in store.root.rglob("*"):
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            assert "nvapi-" not in content
            assert "ghp_" not in content
            assert "x-access-token:" not in content


def test_checkpoint_does_not_call_github_sync():
    """Checkpoint path must not call sync_to_remote automatically."""
    import re

    bootstrap_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "opencode_kaggle"
        / "bootstrap.py"
    ).read_text(encoding="utf-8")
    call_sites = re.findall(r"sync_to_remote\s*\(", bootstrap_src)
    assert call_sites == [], f"found sync_to_remote calls: {call_sites}"
    import_lines = [
        ln for ln in bootstrap_src.splitlines()
        if "import" in ln and "sync_to_remote" in ln
    ]
    assert import_lines == [], f"found imports: {import_lines}"
