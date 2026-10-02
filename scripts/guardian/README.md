# Guardián externo del kernel

El watchdog interno (`src/opencode_cloud/watchdog.py`) reinicia el proceso
de OpenCode **mientras el kernel de Kaggle está vivo**. Cuando la sesión
misma muere (timeout, cuota, inactividad, caída), nada dentro de Kaggle puede
revivirla. Este guardián vive **fuera** de Kaggle y hace lo que hoy hacés a
mano: detectar la sesión muerta y volver a correr el cuaderno vía API.

## Cómo funciona

1. Si hay URL estable configurada (`GUARDIAN_URL`, hostname del túnel
   nombrado de Cloudflare), la prueba primero: si responde, todo está vivo.
2. Si no, pregunta a la API (`kaggle kernels status owner/slug`).
3. Si la sesión está muerta **y** su última corrida empezó hace más de
   `GUARDIAN_MIN_START_AGE` (default 20 min: un arranque normal tarda
   10-15 min entre pip, restore del Dataset y OpenCode), relanza con
   `kaggle kernels push -p <push_dir>`. El cuaderno se auto-arranca
   (instala, restaura el Dataset, levanta OpenCode) y la última celda
   keep-alive mantiene la sesión ejecutando.
4. Por seguridad **no** relanza si el estado es desconocido, si no hay fecha
   de última corrida, o si la corrida empezó hace poco (o está arrancando o
   está muriendo al nacer — relanzar a ciegas quemaría tu cuota de Kaggle).

Al relanzar, el bootstrap restaura el último checkpoint del Dataset
(publicados como mínimo cada 15 min), así que se pierde como máximo el
trabajo posterior al último checkpoint remoto.

## Opción A: GitHub Actions (recomendada, sin PC encendida)

El workflow [kernel-guardian.yml](../../.github/workflows/kernel-guardian.yml)
corre cada 15 minutos. Solo necesita estos Secrets en el repo
(Settings → Secrets → Actions):

| Secret | Valor |
|---|---|
| `KAGGLE_USERNAME` | tu usuario de Kaggle |
| `KAGGLE_KEY` | tu API key (Kaggle → Settings → API → Create New Token) |
| `KAGGLE_KERNEL` | `owner/slug` del cuaderno (ver abajo) |
| `GUARDIAN_URL` | opcional: `https://tu-host.cfargotunnel.com` |

## Opción B: correrlo en tu PC / teléfono (Termux)

```bash
pip install kaggle
export KAGGLE_USERNAME=... KAGGLE_KEY=...
export KAGGLE_KERNEL=owner/slug
# una vez:
python scripts/guardian/guardian.py
# en loop (ej. cada 15 min con cron o un while):
while true; do python scripts/guardian/guardian.py; sleep 900; done
```

## Primera vez: crear el kernel vinculado

El guardián relanza con `kaggle kernels push`, que necesita que el kernel ya
exista con el mismo `id` de [kernel-metadata.json](../../kaggle/kernel-metadata.json):

```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...
# ajustá el "id" en kaggle/kernel-metadata.json a tu owner/slug y:
kaggle kernels push -p kaggle/
```

Ese primer push crea el kernel y lo corre. Desde entonces, el guardián lo
mantiene vivo solo. El `id` también puede venir de `KAGGLE_KERNEL`: el
guardián lo sincroniza al metadata antes de pushear.

> El slug real puede diferir del sugerido (`opencode-continuum-workstation`):
> usá el que te muestre Kaggle tras el primer push y guardalo en el Secret.

## Variables

| Variable | Default | Qué hace |
|---|---|---|
| `KAGGLE_KERNEL` | — (requerida) | `owner/slug` del cuaderno |
| `GUARDIAN_PUSH_DIR` | `kaggle/` del repo | carpeta con el `.ipynb` + metadata |
| `GUARDIAN_URL` | — | URL estable a probar primero |
| `GUARDIAN_MIN_START_AGE` | `1200` (20 min) | edad mínima de la última corrida para relanzar. Subilo si tu arranque tarda más de ~15 min (Dataset pesado); bajalo a `900` si querés recuperación más rápida — a cambio, un kernel que muera al nacer generará más pushes |
| `GUARDIAN_ALLOW_UNKNOWN_TIME` | `0` | en `1`, relanza aunque no haya fecha (no recomendado) |
| `GUARDIAN_TIMEOUT` | — | límite `-t` de la corrida relanzada (segundos) |
| `GUARDIAN_DRY_RUN` | `0` | en `1`, solo dice lo que haría |

Exit codes: `0` vivo / en espera / relanzado OK · `1` falló el push ·
`3` estado desconocido, sin push (miralo vos).
