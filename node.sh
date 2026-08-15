#!/usr/bin/env bash
# Turn this machine into a cluster worker: runs llama.cpp's rpc-server and
# announces itself so the Kestrel head node can find it.
#
#   ./node.sh --mem 8192              donate 8 GB
#   ./node.sh --mem 24576 --cache     donate 24 GB, cache tensors locally
#   ./node.sh --bin ~/llama.cpp/build/bin/rpc-server
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Not installed yet. Run ./install.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
exec python -m kestrel.node "$@"
