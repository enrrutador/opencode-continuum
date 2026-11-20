"""Optional local model server wired into OpenCode Web.

The model is served on localhost (OpenAI-compatible API) and exposed as a
provider in the config file OpenCode actually reads
(``$XDG_CONFIG_HOME/opencode/opencode.jsonc``). A watchdog relaunches
OpenCode so the new provider shows up in the model picker.

Two engines can serve the model:

- ``llama`` (default): llama.cpp ``llama-server`` running a GGUF quant of
  the model on ONE GPU. Small download (~6 GB), starts in about a minute
  and leaves room for a 32k context on a 16 GB T4.
- ``vllm``: vLLM serving the original Hugging Face checkpoint in float16.
  Needs both T4s for the 9B, a long first install and a ~19 GB download.

Environment variables (also supported as Kaggle Secrets):
- LOCAL_MODEL_BASE_URL: use an already-running server instead of an engine
- LOCAL_MODEL_ENGINE: "llama" (default) or "vllm"
- LOCAL_MODEL_GGUF_REPO / LOCAL_MODEL_GGUF_FILE: GGUF source on Hugging Face
- LOCAL_MODEL_NAME: model id served and registered in OpenCode (llama)
- LOCAL_MODEL_CTX: context size for llama-server (default 32768)
- LOCAL_MODEL_ID: Hugging Face id served by vLLM
- LOCAL_MODEL_PROVIDER: provider id in OpenCode (default kaggle-deepseek)
- LOCAL_MODEL_PORT: local server port (default 8000)
- LOCAL_MODEL_DEFAULT: set to 1 to make the local model the default selection
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

DEFAULT_MODEL_ID = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
DEFAULT_GGUF_REPO = "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF"
DEFAULT_GGUF_FILE = "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
DEFAULT_MODEL_NAME = "deepseek-v4-pro-9b-heretic"
DEFAULT_PROVIDER_ID = "kaggle-deepseek"
DEFAULT_LLAMA_BUILD = "b12565"
LLAMA_CUDA = "12.8"
BASE_DIR = Path("/kaggle/working/opencode_cloud")
VLLM_LOG = BASE_DIR / "logs" / "vllm.log"
LLAMA_LOG = BASE_DIR / "logs" / "llama-server.log"
MODELS_DIR = BASE_DIR / "models"
LLAMA_DIR = BASE_DIR / "llama.cpp"


def log(message: str) -> None:
    print(f"[local-model] {message}", flush=True)


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """TCP reachability check without raising."""
    with socket.socket() as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def http_ok(url: str, timeout: float = 3.0) -> bool:
    """True when the URL answers HTTP 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def wait_server(proc: subprocess.Popen, port: int, deadline_s: float, log_path: Path) -> bool:
    """Wait for the server health endpoint; fail fast if the process dies."""
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"el servidor murió al arrancar (código {proc.returncode}); cola del log:")
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in tail[-15:]:
                    log("  " + line)
            except OSError:
                pass
            return False
        if http_ok(f"http://127.0.0.1:{port}/health"):
            return True
        time.sleep(10)
    log(f"el servidor no respondió en {deadline_s:.0f}s (ver {log_path})")
    return False


def download(url: str, dest: Path, timeout: int = 1800) -> Path:
    """Download with curl (resumable); fall back to urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["curl", "-fL", "--retry", "3", "-o", str(dest), url],
            check=True,
            timeout=timeout,
        )
    except Exception:
        log("curl falló; reintentando con urllib...")
        with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    return dest


def ensure_gguf(repo: str, filename: str) -> Path:
    """Download the GGUF once per session (Kaggle disk is ephemeral)."""
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        log(f"GGUF ya descargado: {dest.name}")
        return dest
    try:
        from huggingface_hub import hf_hub_download

        log(f"descargando GGUF {repo}/{filename} ...")
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(MODELS_DIR))
        return Path(path)
    except Exception as exc:
        log(f"huggingface_hub falló ({type(exc).__name__}); bajo con curl...")
        return download(
            f"https://huggingface.co/{repo}/resolve/main/{filename}", dest
        )


def ensure_llama_server() -> tuple[Path, dict]:
    """Return (llama-server binary, extra env). Downloads the CUDA build once."""
    found = sorted(LLAMA_DIR.rglob("llama-server"))
    found = [p for p in found if p.is_file() and os.access(p, os.X_OK)]
    build = os.environ.get("LOCAL_MODEL_LLAMA_BUILD", DEFAULT_LLAMA_BUILD)
    if not found:
        url = (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{build}/llama-{build}-bin-ubuntu-cuda-{LLAMA_CUDA}-x64.tar.gz"
        )
        log(f"descargando llama.cpp {build} (CUDA {LLAMA_CUDA})...")
        tarball = download(url, LLAMA_DIR / "llama.tar.gz")
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(LLAMA_DIR)
        found = [p for p in sorted(LLAMA_DIR.rglob("llama-server")) if p.is_file()]
        if not found:
            raise RuntimeError("el paquete de llama.cpp no trajo llama-server")
        found[0].chmod(0o755)
    server = found[0]
    env = dict(os.environ)
    libs = [str(p.parent) for p in sorted(LLAMA_DIR.rglob("libggml-cuda.so*"))]
    if libs:
        env["LD_LIBRARY_PATH"] = libs[0] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    # Smoke test: if the binary cannot start (missing CUDA runtime libs),
    # fetch the matching cudart package and put its libs on the path.
    probe = subprocess.run(
        [str(server), "--version"], capture_output=True, text=True, env=env, timeout=60
    )
    if probe.returncode != 0:
        log("faltan librerías CUDA: descargo el paquete cudart de llama.cpp...")
        url = (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{build}/cudart-llama-{build}-bin-ubuntu-cuda-{LLAMA_CUDA}-x64.tar.gz"
        )
        tarball = download(url, LLAMA_DIR / "cudart.tar.gz")
        with tarfile.open(tarball) as tf:
            tf.extractall(LLAMA_DIR)
        cudart = [str(p.parent) for p in sorted(LLAMA_DIR.rglob("libcudart.so*"))]
        if cudart:
            env["LD_LIBRARY_PATH"] = cudart[0] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        probe = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, env=env, timeout=60
        )
        if probe.returncode != 0:
            raise RuntimeError("llama-server no arranca: " + (probe.stderr or "")[-300:])
    log("llama-server: " + (probe.stdout or probe.stderr).strip().splitlines()[0])
    return server, env


def llama_server_cmd(server: Path, model_path: Path, port: int, alias: str, ctx: int) -> list:
    return [
        str(server),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99",
        "-c",
        str(ctx),
        "-np",
        "1",
        "--alias",
        alias,
        "--jinja",
    ]


def ensure_llama_model(port: int) -> tuple[str, str]:
    """Serve the GGUF with llama-server. Returns (base_url, served_model_id)."""
    if not gpu_count():
        log("sin GPU y sin LOCAL_MODEL_BASE_URL: se omite el modelo local")
        return "", ""
    repo = os.environ.get("LOCAL_MODEL_GGUF_REPO", DEFAULT_GGUF_REPO)
    filename = os.environ.get("LOCAL_MODEL_GGUF_FILE", DEFAULT_GGUF_FILE)
    alias = os.environ.get("LOCAL_MODEL_NAME", DEFAULT_MODEL_NAME)
    ctx = int(os.environ.get("LOCAL_MODEL_CTX", "32768"))
    model_path = ensure_gguf(repo, filename)
    server, env = ensure_llama_server()
    LLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
    for attempt_ctx in (ctx, 8192):
        cmd = llama_server_cmd(server, model_path, port, alias, attempt_ctx)
        log("arrancando llama-server: " + " ".join(cmd))
        handle = LLAMA_LOG.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        if wait_server(proc, port, 900, LLAMA_LOG):
            log(f"API del modelo lista en puerto {port} (ctx {attempt_ctx})")
            return f"http://127.0.0.1:{port}", alias
        if attempt_ctx == ctx and ctx != 8192:
            log("reintentando con contexto 8192...")
    return "", ""


def ensure_vllm_model(model_id: str, port: int) -> tuple[str, str]:
    """Serve the HF checkpoint with vLLM. Returns (base_url, served_model_id)."""
    gpus = gpu_count()
    if not gpus:
        log("sin GPU y sin LOCAL_MODEL_BASE_URL: se omite el modelo local")
        return "", ""
    try:
        import vllm  # noqa: F401
    except ImportError:
        log("instalando vLLM...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "vllm"],
            check=True,
            timeout=1800,
        )
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--tensor-parallel-size",
        str(max(1, gpus)),
        # The checkpoint is BF16 but the T4 (sm75) cannot do BF16: float16.
        "--dtype",
        "half",
        # CUDA graph capture is slow/fragile on sm75.
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.92",
        "--port",
        str(port),
    ]
    log("arrancando vLLM: " + " ".join(cmd))
    VLLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = VLLM_LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    if wait_server(proc, port, 1800, VLLM_LOG):
        log(f"API del modelo lista en puerto {port}")
        return f"http://127.0.0.1:{port}", model_id
    return "", ""


def ensure_model_server(model_id: str, port: int) -> tuple[str, str]:
    """Return (base_url, served_model_id); start a local engine when possible."""
    external = os.environ.get("LOCAL_MODEL_BASE_URL", "").strip()
    if external:
        log(f"usando servidor existente: {external}")
        return external.rstrip("/"), model_id

    if port_open("127.0.0.1", port) and http_ok(f"http://127.0.0.1:{port}/health"):
        log(f"ya hay un servidor del modelo en el puerto {port}")
        engine = os.environ.get("LOCAL_MODEL_ENGINE", "llama").strip().lower()
        served = (
            os.environ.get("LOCAL_MODEL_NAME", DEFAULT_MODEL_NAME)
            if engine != "vllm"
            else model_id
        )
        return f"http://127.0.0.1:{port}", served

    engine = os.environ.get("LOCAL_MODEL_ENGINE", "llama").strip().lower()
    if engine == "vllm":
        return ensure_vllm_model(model_id, port)
    return ensure_llama_model(port)


def restart_opencode(port: int, timeout: float = 120.0) -> bool:
    """Force a config reload by killing OpenCode; the continuum watchdog relaunches it."""
    subprocess.run(["pkill", "-f", "opencode web"], check=False)
    log("OpenCode detenido; esperando relanzamiento del watchdog...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        if port_open("127.0.0.1", port):
            log("OpenCode relanzado")
            return True
    log("AVISO: OpenCode no respondió tras el reinicio")
    return False


def verify_model(provider_id: str, model_id: str) -> bool:
    """Check that the provider/model is visible to a fresh OpenCode process."""
    try:
        result = subprocess.run(
            ["opencode", "models"], capture_output=True, text=True, timeout=90
        )
    except Exception as exc:
        log(f"no se pudo verificar: {type(exc).__name__}")
        return False
    visible = f"{provider_id}/{model_id}" in result.stdout
    log("modelo seleccionable en la UI: " + ("SÍ" if visible else "NO"))
    if not visible:
        log("salida de opencode models (últimas líneas): " + result.stdout[-500:])
    return visible


def current_tunnel_url(log_name: str = "cloudflare-tunnel.log") -> str:
    """Last quick-tunnel URL from a continuum log (empty if unavailable)."""
    import re

    log_path = BASE_DIR / "logs" / log_name
    if not log_path.exists():
        return ""
    urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_path.read_text())
    return urls[-1] if urls else ""


def load_model_settings() -> None:
    """Promote optional Kaggle Secrets into the environment for this cell."""
    from opencode_cloud.secrets import load_secret

    for key in (
        "LOCAL_MODEL_BASE_URL",
        "LOCAL_MODEL_ENGINE",
        "LOCAL_MODEL_GGUF_REPO",
        "LOCAL_MODEL_GGUF_FILE",
        "LOCAL_MODEL_NAME",
        "LOCAL_MODEL_CTX",
        "LOCAL_MODEL_ID",
        "LOCAL_MODEL_PROVIDER",
        "LOCAL_MODEL_PORT",
        "LOCAL_MODEL_DEFAULT",
    ):
        if not os.environ.get(key):
            value = load_secret(key)
            if value:
                os.environ[key] = str(value)


def install_local_model(*, opencode_port: int = 4096) -> bool:
    """Serve the local 9B model (if possible) and register it in OpenCode."""
    from opencode_cloud.opencode import write_local_model_config
    from opencode_cloud.runtime import get_paths, opencode_config_file

    load_model_settings()
    model_id = os.environ.get("LOCAL_MODEL_ID", DEFAULT_MODEL_ID)
    provider_id = os.environ.get("LOCAL_MODEL_PROVIDER", DEFAULT_PROVIDER_ID)
    port = int(os.environ.get("LOCAL_MODEL_PORT", "8000"))

    base_url, served_id = ensure_model_server(model_id, port)
    if not base_url or not served_id:
        return False

    cfg_path = opencode_config_file(get_paths())
    write_local_model_config(
        cfg_path,
        provider_id=provider_id,
        name="DeepSeek-V4-Pro 9B (local)",
        base_url=base_url,
        models={served_id: "DeepSeek-V4-Pro 9B (heretic)"},
        default_model_id=served_id if os.environ.get("LOCAL_MODEL_DEFAULT") == "1" else None,
    )
    log(f"provider {provider_id} -> {base_url}/v1 en {cfg_path}")

    if not restart_opencode(opencode_port):
        return False
    return verify_model(provider_id, served_id)
