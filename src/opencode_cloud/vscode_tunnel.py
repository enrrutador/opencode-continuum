"""VSCode Tunnel opcional para el runtime Kaggle.

Patrón tomado de kagglevstunnel / kaggle-vscode-remote-tunnel:
  code tunnel --name kaggle --server-data-dir ... --extensions-dir ...

Apagado por defecto (OPENCODE_VSCODE_TUNNEL=1 para activar).
Requiere login previo una vez por runtime:
  ./code tunnel user login --provider github
(no automatizable con token; es device-code interactivo).

Binario y datos viven en /kaggle/working (persistente), no en $HOME.
Solo stdlib + subprocess con argv explícito.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional

VSCODE_CLI_URL = (
    "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
)


def _base_dir() -> Path:
    cand = Path("/kaggle/working/opencode_cloud/bin/vscode-cli")
    try:
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    except OSError:
        pass
    fallback = Path.home() / ".local" / "bin" / "vscode-cli"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _data_dir() -> Path:
    cand = Path("/kaggle/working/.vscode_data")
    try:
        (cand / "cli").mkdir(parents=True, exist_ok=True)
        (cand / "server").mkdir(parents=True, exist_ok=True)
        (cand / "ext").mkdir(parents=True, exist_ok=True)
        return cand
    except OSError:
        pass
    fallback = Path.home() / ".vscode_data"
    (fallback / "cli").mkdir(parents=True, exist_ok=True)
    (fallback / "server").mkdir(parents=True, exist_ok=True)
    (fallback / "ext").mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_vscode_cli(base_dir: Optional[Path] = None) -> str:
    """Descarga VSCode CLI si falta. Idempotente. Retorna path a `code`."""
    dest = Path(base_dir) if base_dir is not None else _base_dir()
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / "code"
    if binary.is_file() and os.access(binary, os.X_OK):
        return str(binary)
    req = urllib.request.Request(
        VSCODE_CLI_URL,
        headers={"User-Agent": "opencode-continuum/5.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if len(data) < 500_000:
        raise RuntimeError("vscode-cli download too small")
    fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz", prefix="vscode_cli_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_bytes(data)
        with tarfile.open(tmp_path, "r:gz") as tf:
            # Extracción contenida: solo el binario `code`.
            members = [m for m in tf.getmembers() if m.name.endswith("/code") or m.name == "code"]
            if not members:
                tf.extractall(dest)
            else:
                for m in members:
                    m.name = "code"
                    tf.extract(m, dest)
        binary.chmod(0o755)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    if not binary.is_file():
        raise RuntimeError("vscode-cli install failed")
    return str(binary)


class VSCodeTunnel:
    """Dueño del proceso `code tunnel`. No se inicia solo."""

    def __init__(
        self,
        *,
        name: str = "kaggle",
        code_bin: Optional[str] = None,
        data_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ):
        self.name = name or "kaggle"
        self.code_bin = code_bin
        self.data_dir = Path(data_dir) if data_dir else _data_dir()
        self.log_path = Path(log_path) if log_path else None
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._output: list[str] = []

    def start(self) -> dict:
        if self._proc is not None and self._proc.poll() is None:
            return {"running": True, "pid": self._proc.pid}
        try:
            binary = self.code_bin or ensure_vscode_cli()
        except Exception as exc:
            return {"running": False, "status": "cli_unavailable", "message": type(exc).__name__}
        server_dir = self.data_dir / "server"
        ext_dir = self.data_dir / "ext"
        server_dir.mkdir(parents=True, exist_ok=True)
        ext_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            binary,
            "tunnel",
            "--accept-server-license-terms",
            "--name",
            self.name,
            "--server-data-dir",
            str(server_dir),
            "--extensions-dir",
            str(ext_dir),
        ]
        log_handle = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.log_path, "a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            if log_handle:
                log_handle.close()
            return {"running": False, "status": "start_failed", "message": type(exc).__name__}

        def reader() -> None:
            stream = self._proc.stdout if self._proc else None
            if stream is None:
                return
            try:
                for line in stream:
                    self._output.append(line)
                    if len(self._output) > 200:
                        self._output = self._output[-100:]
                    if log_handle:
                        log_handle.write(line)
                        log_handle.flush()
            except Exception:
                pass
            finally:
                if log_handle:
                    try:
                        log_handle.close()
                    except Exception:
                        pass

        self._reader = threading.Thread(target=reader, daemon=True)
        self._reader.start()
        pid = self._proc.pid if self._proc else None
        return {
            "running": True,
            "pid": pid,
            "status": "started",
            "message": (
                "Si pide login, corré una vez: "
                "`code tunnel user login --provider github` y reintentá. "
                "En VSCode local: Connect to Tunnel -> kaggle."
            ),
        }

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def status(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        return {
            "running": alive,
            "pid": self._proc.pid if alive and self._proc else None,
            "name": self.name,
        }


def enabled_by_env() -> bool:
    return os.environ.get("OPENCODE_VSCODE_TUNNEL", "0") == "1"


def data_dir_default() -> Path:
    return _data_dir()
