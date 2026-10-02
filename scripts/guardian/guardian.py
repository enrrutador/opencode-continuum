"""External kernel guardian for OpenCode Continuum.

Why this exists: the in-kernel Watchdog (src/opencode_cloud/watchdog.py) can
restart the OpenCode *process* while the Kaggle kernel is alive, but nothing
running inside Kaggle can restart the kernel itself once the session dies
(timeout, quota, preemption). This script lives OUTSIDE Kaggle (GitHub
Actions cron, a PC, a phone with Termux) and presses "Run" for you:

  1. Optionally probes a stable access URL (Cloudflare named tunnel hostname).
     If it answers, the workstation is alive — done.
  2. Otherwise asks the Kaggle API (`kaggle kernels status`) whether the
     kernel session is running.
  3. If the session is dead AND its last run started long enough ago
     (a fresh push needs time to boot; a kernel that dies minutes after
     starting is crash-looping and must NOT be relaunched blindly),
     re-runs the notebook with `kaggle kernels push -p <dir>`.
     The notebook under <dir> must be self-booting (see kaggle/).

Fail-safe defaults: when the kernel state cannot be determined, or the last
run time is unknown, NO push is attempted — burning Kaggle quota on a guess
is worse than waiting for the next check. Exit codes make that visible.

Environment:
  KAGGLE_KERNEL            owner/slug of the workstation notebook (required).
                           The Kaggle CLI also needs KAGGLE_USERNAME/KAGGLE_KEY.
  GUARDIAN_PUSH_DIR        folder with Workstation.ipynb + kernel-metadata.json
                           (default: <repo>/kaggle when run from the repo).
  GUARDIAN_URL             optional stable URL to probe first (named tunnel).
  GUARDIAN_MIN_START_AGE   seconds since last run start before a repush is
                           allowed (default 2700 = 45 min).
  GUARDIAN_ALLOW_UNKNOWN_TIME  if "1", push when dead but last-run time is
                           unknown (default "0" = hold, exit 3).
  GUARDIAN_DRY_RUN         if "1", print what would be done, change nothing.
  GUARDIAN_TIMEOUT         optional `kaggle kernels push -t` limit in seconds.

Exit codes:
  0  alive / recent start (wait) / repushed OK / dry-run
  1  push attempted but failed
  3  state unknown — no push (needs human attention)

Only the standard library is used; the `kaggle` CLI must be on PATH.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

ALIVE_KEYWORDS = ("running", "queued", "active")
DEAD_KEYWORDS = ("complete", "error", "failed", "cancelled", "canceled")
_LAST_RUN_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)")

EXIT_OK = 0
EXIT_PUSH_FAILED = 1
EXIT_UNKNOWN = 3


@dataclass
class GuardianConfig:
    kernel: str
    push_dir: str
    url: str = ""
    min_start_age: int = 2700
    allow_unknown_time: bool = False
    dry_run: bool = False
    timeout: str = ""

    @classmethod
    def from_env(cls) -> "GuardianConfig":
        kernel = os.environ.get("KAGGLE_KERNEL", "").strip()
        here = os.path.dirname(os.path.abspath(__file__))
        default_push_dir = os.path.normpath(os.path.join(here, "..", "..", "kaggle"))
        return cls(
            kernel=kernel,
            push_dir=os.environ.get("GUARDIAN_PUSH_DIR", default_push_dir),
            url=os.environ.get("GUARDIAN_URL", "").strip(),
            min_start_age=int(os.environ.get("GUARDIAN_MIN_START_AGE", "2700")),
            allow_unknown_time=os.environ.get("GUARDIAN_ALLOW_UNKNOWN_TIME") == "1",
            dry_run=os.environ.get("GUARDIAN_DRY_RUN") == "1",
            timeout=os.environ.get("GUARDIAN_TIMEOUT", "").strip(),
        )


def log(msg: str) -> None:
    print(f"[guardian] {msg}", flush=True)


def probe_url(url: str, timeout: float = 15.0) -> bool:
    """True if the stable access URL answers with 2xx/3xx."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception as e:
        log(f"URL probe failed: {type(e).__name__}")
        return False


def classify_status(output: str) -> str:
    """Classify `kaggle kernels status` output: alive | dead | unknown."""
    text = output.lower()
    if any(k in text for k in ALIVE_KEYWORDS):
        return "alive"
    if any(k in text for k in DEAD_KEYWORDS):
        return "dead"
    return "unknown"


def extract_last_run_time(output: str) -> datetime | None:
    """Best-effort parse of the last-run timestamp (assumed UTC)."""
    m = _LAST_RUN_RE.search(output)
    if not m:
        return None
    try:
        dt = datetime.fromisoformat(m.group(1).replace(" ", "T"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def decide(
    status: str,
    last_run: datetime | None,
    now: datetime,
    cfg: GuardianConfig,
) -> tuple[str, str]:
    """Return (action, reason). Actions: ok | wait | push | hold."""
    if status == "alive":
        return "ok", "kernel session is running"
    if status == "unknown":
        return "hold", "could not determine kernel state — no push (fail-safe)"
    # status == "dead"
    if last_run is None:
        if cfg.allow_unknown_time:
            return "push", "kernel dead, last-run time unknown (allowed by config)"
        return "hold", "kernel dead but last-run time unknown — no push (fail-safe)"
    age = (now - last_run).total_seconds()
    if age < 0:
        return "hold", "last-run timestamp is in the future — no push (fail-safe)"
    if age < cfg.min_start_age:
        return (
            "wait",
            f"last run started {int(age)}s ago (< {cfg.min_start_age}s): "
            "still booting or crash-looping — no push",
        )
    return "push", f"last run started {int(age)}s ago — relaunching"


def run_cli(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def sync_metadata_id(push_dir: str, kernel: str, dry_run: bool) -> None:
    """Align kernel-metadata.json id with KAGGLE_KERNEL (the push uses it)."""
    meta_path = os.path.join(push_dir, "kernel-metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as e:
        log(f"cannot read {meta_path}: {e}")
        return
    if meta.get("id") == kernel:
        return
    log(f"metadata id {meta.get('id')!r} -> {kernel!r}")
    if dry_run:
        return
    meta["id"] = kernel
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


def main() -> int:
    cfg = GuardianConfig.from_env()
    if not cfg.kernel or "/" not in cfg.kernel:
        log("KAGGLE_KERNEL must be 'owner/slug'")
        return EXIT_UNKNOWN
    log(f"kernel={cfg.kernel} push_dir={cfg.push_dir} dry_run={cfg.dry_run}")

    if cfg.url:
        log(f"probing stable URL {cfg.url}")
        if probe_url(cfg.url):
            log("stable URL answers — workstation alive")
            return EXIT_OK
        log("stable URL not answering — checking Kaggle API")

    res = run_cli(["kaggle", "kernels", "status", cfg.kernel])
    if res.returncode != 0:
        log(f"`kaggle kernels status` failed: {(res.stderr or res.stdout).strip()[:300]}")
        return EXIT_UNKNOWN
    status = classify_status(res.stdout)
    last_run = extract_last_run_time(res.stdout)
    log(f"api status={status} last_run={last_run}")
    action, reason = decide(status, last_run, datetime.now(timezone.utc), cfg)
    log(f"decision={action}: {reason}")

    if action in ("ok", "wait"):
        return EXIT_OK
    if action == "hold":
        return EXIT_UNKNOWN
    # action == "push"
    if cfg.dry_run:
        log("dry-run: would run `kaggle kernels push`")
        return EXIT_OK
    sync_metadata_id(cfg.push_dir, cfg.kernel, cfg.dry_run)
    cmd = ["kaggle", "kernels", "push", "-p", cfg.push_dir]
    if cfg.timeout:
        cmd += ["-t", cfg.timeout]
    pushed = run_cli(cmd, timeout=300.0)
    out = (pushed.stdout + pushed.stderr).strip()
    log(f"push output: {out[:500]}")
    if pushed.returncode != 0:
        log("push FAILED")
        return EXIT_PUSH_FAILED
    log("push OK — new kernel run queued")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
