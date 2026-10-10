# ============================================================
# OpenCode Continuum — celda v12 (autocontenida)
# Pegá esta celda en un notebook de Kaggle y corréla.
#
# Cambios vs v11:
#   - Picker de sesiones: link primario a la LISTA de sesiones
#     (redirector /go con túnel rápido propio, badge "última" en la
#     pinneada). El deep link a la última sesión queda como secundario.
#   - Sin raises en el bloque de links: si no hay URL pública aún,
#     registra los paths relativos y sigue con el keep-alive.
#   - Un solo repo como fuente de verdad (opencode-continuum@main).
#   - Keep-alive incluido: mantiene la sesión viva y cede al
#     guardián externo si OpenCode no se recupera.
# ============================================================
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def log(*a, **k):
    print(*a, flush=True, **k)


def run(cmd, check=True, **kw):
    log(" $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, **kw)


REPO = Path("/kaggle/working/opencode-continuum")
WORKSPACE = Path("/kaggle/working/opencode_cloud/workspace")
METADATA = Path("/kaggle/working/opencode_cloud/metadata")
DATASET_ID = os.environ.get("OPENCODE_CLOUD_DATASET", "")
PORT = 4096

log("=" * 60)
log("OpenCode Continuum v12")
log("=" * 60)


# --- secrets ---
try:
    from kaggle_secrets import UserSecretsClient

    usc = UserSecretsClient()
    for key in (
        "NVIDIA_API_KEY",
        "OPENCODE_CLOUD_DATASET",
        "OPENCODE_SERVER_PASSWORD",
        "OPENCODE_SERVER_USERNAME",
    ):
        try:
            val = usc.get_secret(key)
            if val:
                os.environ[key] = val
                if key == "OPENCODE_CLOUD_DATASET":
                    DATASET_ID = val
        except Exception:
            pass
except Exception as e:
    log("secrets:", type(e).__name__)

if not DATASET_ID:
    raise RuntimeError("Falta OPENCODE_CLOUD_DATASET ('usuario/dataset')")
os.environ["OPENCODE_CLOUD_DATASET"] = DATASET_ID
log("Dataset:", DATASET_ID)


# --- deps ---
for mod, pkg in (("kagglehub", "kagglehub"), ("requests", "requests")):
    try:
        __import__(mod)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", pkg], timeout=300)


# --- repo: única fuente de verdad (main, CI verde) ---
if not (REPO / "src/opencode_kaggle/bootstrap.py").exists():
    if REPO.exists():
        shutil.rmtree(REPO)
    run(
        ["git", "clone", "--depth", "1",
         "https://github.com/enrrutador/opencode-continuum.git", str(REPO)],
        timeout=180,
    )
else:
    run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False, timeout=120)

sys.path.insert(0, str(REPO / "src"))


WORKSPACE.mkdir(parents=True, exist_ok=True)
if not (WORKSPACE / ".git").exists():
    run(["git", "init"], cwd=str(WORKSPACE), check=False)
    run(["git", "-C", str(WORKSPACE), "config", "user.email", "opencode@kaggle.local"], check=False)
    run(["git", "-C", str(WORKSPACE), "config", "user.name", "OpenCode"], check=False)
    (WORKSPACE / "README.md").write_text("# OpenCode workspace\n", encoding="utf-8")
    run(["git", "-C", str(WORKSPACE), "add", "-A"], check=False)
    run(["git", "-C", str(WORKSPACE), "commit", "-m", "init"], check=False)


# --- bootstrap ---
log("\n[bootstrap]")
from opencode_cloud.checkpoint import CheckpointPolicy
from opencode_kaggle.bootstrap import bootstrap


policy = CheckpointPolicy(local_interval=120, min_publish_interval=900)
info = bootstrap(dataset_id=DATASET_ID, opencode_port=PORT, policy=policy, enable_access_layer=True)
log(json.dumps({k: v for k, v in info.items() if k != "shutdown"}, indent=2, default=str))
if not info.get("ok"):
    raise RuntimeError(f"Bootstrap falló: {info}")

recovery = info.get("recovery", "")


# --- URL pública (tolerante: nunca mata el keep-alive) ---
wa = info.get("web_access") or {}
base_url = (wa.get("url") or "").rstrip("/")
if not base_url:
    lp = Path("/kaggle/working/opencode_cloud/logs/cloudflare-tunnel.log")
    if lp.exists():
        m = re.search(r"https://[a-zA-Z0-9.-]+\.(?:trycloudflare|cfargotunnel)\.com[^\s]*", lp.read_text())
        if m:
            base_url = m.group(0).rstrip("/")
named = wa.get("provider") == "cloudflare_named_tunnel"

ws_b64 = base64.urlsafe_b64encode(str(WORKSPACE).encode()).decode().rstrip("=")
PROJECT_URL = f"{base_url}/{ws_b64}" if base_url else f"/{ws_b64}"
SESSION_URL = f"{base_url}/{ws_b64}/session" if base_url else f"/{ws_b64}/session"

# Picker: túnel rápido dedicado (lo expone el bootstrap) o regla /go
# del túnel nombrado. La raíz del picker también sirve (/ == /go).
sp = info.get("session_picker") or {}
if sp.get("available") and sp.get("url"):
    PICKER_URL = sp["url"].rstrip("/")
elif base_url:
    PICKER_URL = base_url + "/go"
else:
    PICKER_URL = "/go"


# --- API local de OpenCode ---
def api(method, path, data=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    pw = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
    user = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    if pw:
        tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode() or "null")


# --- sesión: pin persistente + última activa ---
from opencode_cloud.entry import load_pin, pick_session, save_pin

pin_path = METADATA / "session.json"
q = "?directory=" + urllib.parse.quote(str(WORKSPACE))
sessions = []
try:
    sessions = api("GET", "/session" + q) or []
    if not isinstance(sessions, list):
        sessions = []
except Exception as e:
    log("GET /session:", type(e).__name__, e)

if not sessions:
    try:
        s = api("POST", "/session" + q, b"{}")
        if s and s.get("id"):
            sessions = [s]
            log("Creada sesión seed:", s.get("id"), s.get("title"))
    except Exception as e:
        log("POST /session:", type(e).__name__, e)

chosen = pick_session(sessions, load_pin(pin_path))
if chosen:
    OPEN_URL = f"{base_url}/{ws_b64}/session/{chosen['id']}" if base_url else f"/{ws_b64}/session/{chosen['id']}"
    OPEN_TITLE = chosen.get("title") or chosen["id"]
    METADATA.mkdir(parents=True, exist_ok=True)
    save_pin(pin_path, chosen["id"], OPEN_TITLE)
else:
    OPEN_URL = SESSION_URL
    OPEN_TITLE = "(nueva conversación)"


# --- links ---
out = Path("/kaggle/working/opencode_cloud/OPEN_THIS_URL.txt")
lines = [
    f"ELEGIR SESION: {PICKER_URL}",
    f"ULTIMA SESION: {OPEN_URL}",
    "",
    f"sesion: {OPEN_TITLE}",
    f"proyecto: {PROJECT_URL}",
    f"recovery: {recovery}",
    f"sesiones: {len(sessions)}",
]
if named:
    lines.append("bookmark_permanente: /go (ver README del repo)")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")

log("")
log("######## ELEGIR SESIÓN (lista, una sola línea) ########")
print(PICKER_URL, flush=True)
log("### o continuá directamente la última ###")
print(OPEN_URL, flush=True)
log("#########################################")
log(f"Sesión: {OPEN_TITLE} | historial: {len(sessions)} sesiones")
if not base_url:
    log("(sin URL pública aún: paths relativos registrados; esperá el túnel o configurá uno nombrado)")
if named:
    log("Túnel nombrado: bookmark permanente en", PICKER_URL, "(regla /go, ver README del repo).")
log("Workspace:", WORKSPACE)
log("Recovery:", recovery)
log("Links también en:", out)

OPEN_CODE_URL = OPEN_URL
OPEN_CODE_PROJECT_URL = PROJECT_URL
WORKSTATION_RESULT = info


# --- keep-alive ---
log("")
log("[keep-alive] manteniendo la sesión viva (Interrumpir celda para frenar)")
misses = 0
ticks = 0
while True:
    s = socket.socket()
    s.settimeout(5)
    try:
        s.connect(("127.0.0.1", PORT))
        alive = True
    except OSError:
        alive = False
    finally:
        s.close()
    ticks += 1
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if alive:
        misses = 0
        if ticks % 10 == 1:
            log(f"[keep-alive] {now} OpenCode responde en {PORT}")
    else:
        misses += 1
        log(f"[keep-alive] {now} puerto {PORT} sin respuesta ({misses}/5)")
        if misses > 5:
            log("[keep-alive] sin recuperación: termino; el guardián externo relanza el kernel.")
            break
    time.sleep(60)
