"""Optional local model server wired into OpenCode Web.

The model is served by vLLM on localhost (OpenAI-compatible API) and exposed
as a provider in the config file OpenCode actually reads
(``$XDG_CONFIG_HOME/opencode/opencode.jsonc``). A watchdog relaunches OpenCode
so the new provider shows up in the model picker.

Environment variables (also supported as Kaggle Secrets):
- LOCAL_MODEL_BASE_URL: use an already-running server instead of starting vLLM
- LOCAL_MODEL_ID: Hugging Face id served by vLLM
- LOCAL_MODEL_PROVIDER: provider id in OpenCode (default kaggle-deepseek)
- LOCAL_MODEL_PORT: local vLLM port (default 8000)
- LOCAL_MODEL_DEFAULT: set to 1 to make the local model the default selection
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL_ID = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
DEFAULT_PROVIDER_ID = "kaggle-deepseek"
VLLM_LOG = Path("/kaggle/working/opencode_cloud/logs/vllm.log")


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


def ensure_model_server(model_id: str, port: int) -> str:
    """Return base URL for the model API; start vLLM locally when possible."""
    external = os.environ.get("LOCAL_MODEL_BASE_URL", "").strip()
    if external:
        log(f"usando servidor existente: {external}")
        return external.rstrip("/")

    if not port_open("127.0.0.1", port):
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            gpu_count = 0
        if not gpu_count:
            log("sin GPU y sin LOCAL_MODEL_BASE_URL: se omite el modelo local")
            return ""

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
            str(max(1, gpu_count)),
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
        subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(15)
            if port_open("127.0.0.1", port):
                break
        else:
            log(f"vLLM no respondió en 30 min (ver {VLLM_LOG})")
            return ""

    log(f"API del modelo lista en puerto {port}")
    return f"http://127.0.0.1:{port}"


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

    log_path = Path("/kaggle/working/opencode_cloud/logs") / log_name
    if not log_path.exists():
        return ""
    urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_path.read_text())
    return urls[-1] if urls else ""


def load_model_settings() -> None:
    """Promote optional Kaggle Secrets into the environment for this cell."""
    from opencode_cloud.secrets import load_secret

    for key in (
        "LOCAL_MODEL_BASE_URL",
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

    base_url = ensure_model_server(model_id, port)
    if not base_url:
        return False

    cfg_path = opencode_config_file(get_paths())
    write_local_model_config(
        cfg_path,
        provider_id=provider_id,
        name="DeepSeek-V4-Pro 9B (local)",
        base_url=base_url,
        models={model_id: "DeepSeek-V4-Pro 9B (heretic)"},
        default_model_id=model_id if os.environ.get("LOCAL_MODEL_DEFAULT") == "1" else None,
    )
    log(f"provider {provider_id} -> {base_url}/v1 en {cfg_path}")

    if not restart_opencode(opencode_port):
        return False
    return verify_model(provider_id, model_id)
