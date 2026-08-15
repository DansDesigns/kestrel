#!/usr/bin/env bash
# Launch Kestrel. Any arguments are passed through (--cli, --url, ...).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Not installed yet. Run ./install.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
exec python -m kestrel "$@"
