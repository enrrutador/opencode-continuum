"""Persistence layer for OpenCode Cloud Workstation on Kaggle.

See module docstring in repo history for architecture overview.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class RecoveryStatus(str, Enum):
    RESTORED_FROM_DATASET = "RESTORED_FROM_DATASET"
    FRESH_WORKSTATION = "FRESH_WORKSTATION"
    RESTORE_FAILED = "RESTORE_FAILED"


@dataclass
class RecoveryResult:
    status: RecoveryStatus
    dataset_id: str = ""
    message: str = ""
    details: Optional[dict] = None


class DownloadErrorKind(str, Enum):
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_DATASET = "INVALID_DATASET"
    DOWNLOAD_ERROR = "DOWNLOAD_ERROR"


def classify_download_error(exc: BaseException) -> DownloadErrorKind:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if any(t in msg for t in ("401", "unauthorized", "authentication", "invalid credentials", "api key")):
        return DownloadErrorKind.AUTHENTICATION_ERROR
    if any(t in msg for t in ("403", "forbidden", "permission denied", "not allowed")):
        return DownloadErrorKind.AUTHORIZATION_ERROR
    if any(t in msg for t in ("429", "rate limit", "too many requests")):
        return DownloadErrorKind.RATE_LIMIT
    if any(t in msg for t in ("timeout", "timed out", "connection", "network", "dns", "unreachable", "ssl")):
        return DownloadErrorKind.NETWORK_ERROR
    if any(t in msg for t in ("404", "not found", "does not exist", "dataset does not exist", "no such dataset")):
        return DownloadErrorKind.DATASET_NOT_FOUND
    if "not found" in name or "http404" in name:
        return DownloadErrorKind.DATASET_NOT_FOUND
    return DownloadErrorKind.DOWNLOAD_ERROR


# Archivos/dirs que nunca deben publicarse al Dataset:
# - node_modules y caches: bloat (302MB vs 20MB) y rompen el manifest (10k files truncated)
# - .git/: Kaggle lo strippea al descargar -> integrity check falla con "missing"
# - WAL/SHM de sqlite: si se copian a medias corrompen; se hace checkpoint antes
SKIP_DIR_NAMES = frozenset(
    {"node_modules", "__pycache__", ".cache", "cache", ".git", ".ipynb_checkpoints"}
)


def _should_skip_for_manifest(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    # SQLite WAL/SHM nunca van al manifest (se hace checkpoint antes de copiar)
    if rel_posix.endswith("-wal") or rel_posix.endswith("-shm"):
        return True
    if rel_posix.endswith(".pyc"):
        return True
    return False


def sqlite_checkpoint_db(db_path: Path) -> bool:
    """Fuerza WAL -> DB para que el .db solo ya tenga todas las sesiones.

    Sin esto, copiar opencode.db sin su WAL pierde mensajes recientes.
    Retorna True si hizo checkpoint.
    """
    try:
        import sqlite3

        db_path = Path(db_path)
        if not db_path.is_file():
            return False
        con = sqlite3.connect(str(db_path), timeout=10)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.commit()
        finally:
            con.close()
        return True
    except Exception:
        return False


def unify_opencode_sessions_to_latest_project(opencode_db: Path) -> dict:
    """Mueve sesiones huerfanas al proyecto mas reciente del mismo worktree.

    Causa del bug "veo sesiones pero vacias": cada bootstrap fresco crea
    un project_id nuevo aleatorio para el mismo /workspace. El API
    GET /session solo lista el proyecto actual, las viejas quedan ocultas.
    Esta funcion las re-asigna al proyecto mas nuevo para que reaparezcan.
    Retorna {"migrated": n, "from": old_id, "to": new_id}.
    """
    try:
        import sqlite3

        opencode_db = Path(opencode_db)
        if not opencode_db.is_file():
            return {"migrated": 0, "error": "db_missing"}
        con = sqlite3.connect(str(opencode_db), timeout=10)
        try:
            cur = con.cursor()
            cur.execute("SELECT id FROM project ORDER BY rowid DESC LIMIT 10")
            projects = [r[0] for r in cur.fetchall()]
            if len(projects) < 2:
                return {"migrated": 0, "projects": len(projects)}
            latest = projects[0]
            migrated = 0
            for old in projects[1:]:
                cur.execute(
                    "UPDATE session SET project_id=? WHERE project_id=?",
                    (latest, old),
                )
                migrated += cur.rowcount or 0
            con.commit()
            return {"migrated": migrated, "to": latest, "projects": len(projects)}
        finally:
            con.close()
    except Exception as e:
        return {"migrated": 0, "error": f"{type(e).__name__}: {e}"}


def build_integrity_manifest(
    root: Path,
    *,
    checkpoint_id: str = "",
    max_hash_bytes: int = 2 * 1024 * 1024,
    max_files: int = 10_000,
) -> dict:
    root = Path(root)
    files = []
    incomplete = False
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "manifest.json" or p.is_symlink():
            continue
        try:
            rel = p.relative_to(root).as_posix()
            if _should_skip_for_manifest(rel):
                continue
            st = p.stat()
            entry = {
                "path": rel,
                "size": st.st_size,
                "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
                "sha256": None,
            }
            if st.st_size <= max_hash_bytes:
                h = hashlib.sha256()
                with p.open("rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
                entry["sha256"] = h.hexdigest()
            files.append(entry)
            if len(files) >= max_files:
                incomplete = True
                break
        except OSError:
            continue
    return {
        "schema_version": "5.0.0",
        "kind": "opencode-cloud-workstation-manifest",
        "checkpoint_id": checkpoint_id or str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "incomplete": incomplete,
        "files": files,
    }


def validate_integrity_manifest(root: Path, manifest: Optional[dict] = None) -> dict:
    root = Path(root)
    if manifest is None:
        mpath = root / "manifest.json"
        if not mpath.exists():
            return {"ok": True, "status": "no_manifest", "message": "no manifest present (legacy)"}
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "status": "invalid_manifest", "message": f"manifest unreadable: {e}"}
    if not isinstance(manifest, dict) or manifest.get("kind") != "opencode-cloud-workstation-manifest":
        return {"ok": False, "status": "invalid_manifest", "message": "bad manifest kind"}
    mismatches = []
    for entry in manifest.get("files") or []:
        rel = entry.get("path")
        if not rel:
            continue
        # Compat: manifests viejos incluian workspace/.git/* pero Kaggle
        # strippea .git al descargar. No debe tumbar el restore.
        if "/.git/" in f"/{rel}" or rel.startswith(".git/") or rel == ".git":
            continue
        target = root / rel
        if not target.is_file():
            mismatches.append({"path": rel, "error": "missing"})
            continue
        try:
            st = target.stat()
            if st.st_size != entry.get("size"):
                mismatches.append({"path": rel, "error": "size_mismatch"})
                continue
            expected_hash = entry.get("sha256")
            if expected_hash:
                h = hashlib.sha256()
                with target.open("rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
                if h.hexdigest() != expected_hash:
                    mismatches.append({"path": rel, "error": "hash_mismatch"})
        except OSError as e:
            mismatches.append({"path": rel, "error": str(e)})
    if mismatches:
        return {
            "ok": False,
            "status": "integrity_failed",
            "message": f"{len(mismatches)} file(s) failed integrity check",
            "mismatches": mismatches[:20],
        }
    return {"ok": True, "status": "valid", "message": "manifest ok"}


class PersistentStore:
    MARKER_FILE = "workstation.json"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.workspace = self.root / "workspace"
        self.state = self.root / "state"
        self.config = self.root / "config"
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.metadata_dir = self.root / "metadata"
        self.staging = self.root / "staging"

    def ensure_structure(self) -> None:
        for d in (
            self.root, self.workspace, self.state, self.config,
            self.checkpoints, self.logs, self.metadata_dir, self.staging,
            self.state / "opencode", self.state / "config",
        ):
            d.mkdir(parents=True, exist_ok=True)

    def is_valid_workstation(self, path: Optional[Path] = None) -> bool:
        return self.validate_workstation(path).get("ok", False)

    def validate_workstation(self, path: Optional[Path] = None) -> dict:
        base = Path(path) if path is not None else self.root
        marker = base / self.MARKER_FILE
        if not marker.exists():
            return {"ok": False, "status": "incomplete", "message": "missing workstation.json marker"}
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "status": "invalid", "message": f"marker unreadable: {type(e).__name__}"}
        if not isinstance(data, dict):
            return {"ok": False, "status": "invalid", "message": "marker is not an object"}
        if data.get("kind") != "opencode-cloud-workstation":
            return {"ok": False, "status": "incompatible", "message": f"unexpected kind: {data.get('kind')!r}"}
        version = str(data.get("version", ""))
        if not version.startswith("5."):
            return {"ok": False, "status": "incompatible", "message": f"incompatible schema version: {version!r}"}
        if not (base / "workspace").is_dir():
            return {"ok": False, "status": "incomplete", "message": "missing workspace/ directory"}
        if not ((base / "state").is_dir() or (base / "config").is_dir()):
            return {"ok": False, "status": "incomplete", "message": "missing state/ and config/"}
        return {"ok": True, "status": "valid", "message": "workstation ok", "version": version}

    def write_marker(self, extra: Optional[dict] = None) -> None:
        data = {
            "kind": "opencode-cloud-workstation",
            "version": "5.0.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            data.update(extra)
        self.ensure_structure()
        (self.root / self.MARKER_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_local(self, *, opencode_data: Path, opencode_config: Path, workspace: Path, xdg_root: Optional[Path] = None, extra_meta: Optional[dict] = None) -> dict:
        self.ensure_structure()

        # Flush WAL -> DB antes de copiar: si no, el .db copiado pierde
        # los mensajes recientes que aun estan solo en .db-wal.
        try:
            opencode_data = Path(opencode_data)
            for db_path in [opencode_data / "opencode.db"] + list(
                opencode_data.rglob("opencode.db")
            ):
                sqlite_checkpoint_db(db_path)
                break
        except Exception:
            pass

        def _copy(src: Path, dst: Path) -> None:
            if not src.exists():
                return
            src, dst = Path(src), Path(dst)
            try:
                if dst.exists() and src.resolve() == dst.resolve():
                    return
            except Exception:
                pass
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

        _copy(opencode_data, self.state / "opencode")
        _copy(opencode_config, self.state / "config")
        _copy(opencode_config, self.config)
        _copy(workspace, self.workspace)
        if xdg_root is not None:
            _copy(xdg_root, self.state / "xdg")
        meta = {"saved_at": datetime.now(timezone.utc).isoformat(), "kind": "local_checkpoint"}
        if extra_meta:
            meta.update(extra_meta)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        (self.metadata_dir / "last_local.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.write_marker(extra_meta)
        return meta

    def restore_to(self, *, opencode_data: Path, opencode_config: Path, workspace: Path, xdg_root: Optional[Path] = None, source: Optional[Path] = None) -> bool:
        base = Path(source) if source is not None else self.root
        restored_any = False

        def _restore(src: Path, dst: Path) -> bool:
            if not src.exists():
                return False
            src, dst = Path(src), Path(dst)
            try:
                if dst.exists() and src.resolve() == dst.resolve():
                    return True
            except Exception:
                pass
            try:
                if src.is_dir() and not any(src.iterdir()):
                    return False
            except Exception:
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return True

        # Canonical source for live xdg is state/xdg (it already contains
        # share/opencode/opencode.db). Legacy datasets may only have
        # state/opencode. Restore xdg FIRST, then fill opencode_data from the
        # legacy location ONLY if xdg did not already provide it. The old
        # order (opencode first, xdg second) silently overwrote the first
        # copy, so whichever was stale won depending on timing.
        xdg_restored_opencode_db = False
        if xdg_root is not None:
            # Prefer new location state/xdg, fallback to legacy xdg at root
            xdg_src = base / "state" / "xdg"
            if not xdg_src.exists():
                xdg_src = base / "xdg"
            if _restore(xdg_src, xdg_root):
                restored_any = True
                try:
                    xdg_restored_opencode_db = (Path(xdg_root) / "share" / "opencode" / "opencode.db").is_file()
                except Exception:
                    xdg_restored_opencode_db = False
        if not xdg_restored_opencode_db:
            if _restore(base / "state" / "opencode", opencode_data):
                restored_any = True
        cfg_src = base / "state" / "config"
        if not cfg_src.exists():
            cfg_src = base / "config"
        if _restore(cfg_src, opencode_config):
            restored_any = True
        if _restore(base / "workspace", workspace):
            restored_any = True
        return restored_any

    def prepare_staging(self) -> Path:
        self.ensure_structure()
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=True)

        def _ignore(_dir: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            for n in names:
                if n in SKIP_DIR_NAMES:
                    ignored.add(n)
                elif n.endswith("-wal") or n.endswith("-shm"):
                    ignored.add(n)
                elif n.endswith(".pyc"):
                    ignored.add(n)
                elif n == "manifest.json":
                    ignored.add(n)
            return ignored

        # NOTE: do NOT stage live root/xdg. The backup copy state/xdg already
        # contains it; staging both duplicated opencode.db (3x) and let
        # live/stale copies diverge. Stage only backup + workspace.
        for name in ("workspace", "state", "config", "checkpoints", "logs", "metadata"):
            src = self.root / name
            if src.exists():
                shutil.copytree(
                    src, self.staging / name, dirs_exist_ok=True, ignore=_ignore
                )
        marker_src = self.root / self.MARKER_FILE
        if marker_src.exists():
            shutil.copy2(marker_src, self.staging / self.MARKER_FILE)
        else:
            data = {"kind": "opencode-cloud-workstation", "version": "5.0.0", "staged_at": datetime.now(timezone.utc).isoformat()}
            (self.staging / self.MARKER_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            manifest = build_integrity_manifest(self.staging)
            (self.staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass
        return self.staging

    def never_writes_to_kaggle_datasets(self) -> bool:
        try:
            return not str(self.root.resolve()).startswith("/kaggle/datasets")
        except Exception:
            return not str(self.root).startswith("/kaggle/datasets")


class KagglePersistence:
    def __init__(self, dataset_id: str, working_root: Path):
        if not dataset_id or "/" not in dataset_id:
            raise ValueError("dataset_id must be 'owner/dataset'")
        if "/versions/" in dataset_id:
            raise ValueError("dataset_id must not include a version for upload")
        self.dataset_id = dataset_id
        self.working_root = Path(working_root)
        self.download_cache = self.working_root / "opencode_cloud_download"
        self.download_cache.mkdir(parents=True, exist_ok=True)

    def _kagglehub(self):
        try:
            import kagglehub  # type: ignore
            return kagglehub
        except ImportError as e:
            raise RuntimeError("kagglehub is required for Kaggle Dataset persistence. Install with: pip install kagglehub") from e

    def download(self, force: bool = False) -> tuple[bool, Optional[Path], str]:
        kh = self._kagglehub()
        try:
            try:
                path_str = kh.dataset_download(self.dataset_id, output_dir=str(self.download_cache), force_download=force)
            except TypeError:
                path_str = kh.dataset_download(self.dataset_id, force_download=force)
            path = Path(path_str)
            if path.exists():
                return True, path, f"Downloaded to {path}"
            return False, None, f"Download returned non-existent path: {path_str}"
        except Exception as e:
            kind = classify_download_error(e)
            if kind == DownloadErrorKind.DATASET_NOT_FOUND:
                return False, None, f"Dataset does not exist yet: {self.dataset_id}"
            return False, None, f"{kind.value}: {e}"

    def upload(self, local_dir: Path, version_notes: str = "") -> tuple[bool, str]:
        local_dir = Path(local_dir)
        if not local_dir.exists() or not any(local_dir.iterdir()):
            return False, "Staging directory is empty or missing"
        kh = self._kagglehub()
        try:
            kh.dataset_upload(self.dataset_id, str(local_dir), version_notes=version_notes or "OpenCode Cloud Workstation checkpoint")
            return True, f"Published to {self.dataset_id}"
        except Exception as e:
            return False, f"Upload failed: {e}"

    def recover_into(self, store: PersistentStore) -> RecoveryResult:
        ok, download_path, msg = self.download()
        if not ok or download_path is None:
            upper = msg.upper()
            if "DATASET DOES NOT EXIST" in upper or upper.startswith("DATASET_NOT_FOUND"):
                return RecoveryResult(status=RecoveryStatus.FRESH_WORKSTATION, dataset_id=self.dataset_id, message=msg)
            for kind in DownloadErrorKind:
                if upper.startswith(kind.value) or kind.value in upper:
                    if kind == DownloadErrorKind.DATASET_NOT_FOUND:
                        return RecoveryResult(status=RecoveryStatus.FRESH_WORKSTATION, dataset_id=self.dataset_id, message=msg)
                    return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message=msg, details={"error_kind": kind.value})
            return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message=msg)

        if not store.is_valid_workstation(download_path):
            try:
                has_files = any(download_path.rglob("*"))
            except Exception:
                has_files = False
            if not has_files:
                return RecoveryResult(status=RecoveryStatus.FRESH_WORKSTATION, dataset_id=self.dataset_id, message="Dataset exists but is empty; starting fresh")
            return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message="Downloaded Dataset does not contain a valid workstation marker", details={"path": str(download_path)})

        try:
            snapshot = self.working_root / "opencode_cloud_restore_snapshot"
            if snapshot.exists():
                shutil.rmtree(snapshot)
            shutil.copytree(download_path, snapshot)
            if store.root.exists():
                for child in list(store.root.iterdir()):
                    if child.name in ("staging",):
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            store.ensure_structure()
            for item in snapshot.iterdir():
                dest = store.root / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            if not store.is_valid_workstation():
                return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message="Copy completed but workstation validation failed")
            integrity = validate_integrity_manifest(store.root)
            if not integrity.get("ok") and integrity.get("status") != "no_manifest":
                return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message=f"Integrity check failed: {integrity.get('message')}", details=integrity)
            return RecoveryResult(status=RecoveryStatus.RESTORED_FROM_DATASET, dataset_id=self.dataset_id, message=f"Restored from {download_path}", details={"path": str(download_path)})
        except Exception as e:
            return RecoveryResult(status=RecoveryStatus.RESTORE_FAILED, dataset_id=self.dataset_id, message=f"Restore copy failed: {e}")

    def publish_from_store(self, store: PersistentStore, version_notes: str = "") -> tuple[bool, str]:
        validation = store.validate_workstation()
        if not validation.get("ok"):
            return False, f"refusing to publish invalid workstation: {validation.get('message')}"
        staging = store.prepare_staging()
        return self.upload(staging, version_notes=version_notes)


def default_store(working: Optional[Path] = None) -> PersistentStore:
    if working is None:
        from .runtime import get_paths
        working = get_paths().working
    root = Path(working) / "opencode_cloud"
    store = PersistentStore(root)
    store.ensure_structure()
    return store


def default_kaggle_persistence(dataset_id: Optional[str] = None, working: Optional[Path] = None) -> KagglePersistence:
    import os
    dataset_id = dataset_id or os.environ.get("OPENCODE_CLOUD_DATASET")
    if not dataset_id:
        raise RuntimeError("OPENCODE_CLOUD_DATASET not configured (expected 'owner/dataset')")
    if working is None:
        from .runtime import get_paths
        working = get_paths().working
    return KagglePersistence(dataset_id, Path(working))
