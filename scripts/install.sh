#!/usr/bin/env bash
# Install runtime dependencies inside a Kaggle kernel (or similar Linux env).
# Security: never pipe remote content into a shell.
# Node install matches src/opencode_cloud/opencode.py strategy:
#   prefer existing node → else download NodeSource setup to a temp file →
#   execute that local file with explicit argv → apt-get install nodejs → cleanup.
set -euo pipefail

apt-get update -qq
apt-get install -y -qq git curl jq rsync ca-certificates

if ! command -v node >/dev/null 2>&1; then
  SETUP_TMP="$(mktemp /tmp/nodesource_XXXXXX.sh)"
  cleanup() { rm -f "$SETUP_TMP"; }
  trap cleanup EXIT
  # Download with HTTP failure detection; never remote-pipe install
  curl -fsSL --fail --max-time 60 \
    -A "opencode-continuum/5.0" \
    -o "$SETUP_TMP" \
    "https://deb.nodesource.com/setup_22.x"
  chmod 700 "$SETUP_TMP"
  bash "$SETUP_TMP"
  apt-get install -y nodejs
  cleanup
  trap - EXIT
fi

if ! command -v opencode >/dev/null 2>&1; then
  npm install -g opencode-ai
fi

python -m pip install -q --upgrade pip
python -m pip install -q kagglehub

echo "OK node=$(node --version) opencode=$(opencode --version 2>/dev/null || echo pending)"
