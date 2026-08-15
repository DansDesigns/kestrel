#!/usr/bin/env bash
# Kestrel installer (Linux, macOS, WSL)
#
# Run with no arguments and it will ask what you want; every choice below has a
# sensible default, so pressing Enter throughout is a valid way to install.
#
#   bash install.sh                            interactive
#   bash install.sh --no-llama                 dependencies only, ask nothing
#   bash install.sh --llama                    install llama.cpp, auto backend
#   bash install.sh --llama --backend vulkan   choose the accelerator
#   bash install.sh --llama --source           build from source
#   bash install.sh --llama --no-rpc           omit the RPC backend (no clustering)
#   bash install.sh --reinstall                remove Kestrel's copy and install again
#   bash install.sh --uninstall-llama          remove Kestrel's copy of llama.cpp
#   bash install.sh --speech                   also install offline speech engines
#   bash install.sh --no-speech                skip them without asking
#   bash install.sh --yes                      accept every default, no prompts
#
# Backends: auto, cpu, cuda, vulkan, hip, metal, sycl.
set -euo pipefail
cd "$(dirname "$0")"

INSTALL_LLAMA=""      # "", 1, or 0 — empty means "ask"
BACKEND=""
SOURCE=""
WITH_RPC=""
REMOVE_FIRST=0
INSTALL_SPEECH=""
ASSUME_YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --llama|--llama.cpp)  INSTALL_LLAMA=1 ;;
    --no-llama|--skip-llama) INSTALL_LLAMA=0 ;;
    --backend)  BACKEND="${2:-auto}"; INSTALL_LLAMA=1; shift ;;
    --backend=*) BACKEND="${1#*=}"; INSTALL_LLAMA=1 ;;
    --source|--build) SOURCE=1; INSTALL_LLAMA=1 ;;
    --prebuilt) SOURCE=0 ;;
    --no-rpc)   WITH_RPC=0 ;;
    --rpc)      WITH_RPC=1 ;;
    --reinstall) INSTALL_LLAMA=1; REMOVE_FIRST=1 ;;
    --uninstall-llama) INSTALL_LLAMA=2 ;;
    --speech)    INSTALL_SPEECH=1 ;;
    --no-speech) INSTALL_SPEECH=0 ;;
    -y|--yes)   ASSUME_YES=1 ;;
    -h|--help)  sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

if [ -n "$BACKEND" ]; then
  case "$BACKEND" in
    auto|cpu|cuda|vulkan|hip|metal|sycl) ;;
    *) echo "unknown backend: $BACKEND (auto, cpu, cuda, vulkan, hip, metal, sycl)" >&2
       exit 1 ;;
  esac
fi

# Prompts only make sense on a terminal. Piped or scripted runs fall through to
# the flags and their defaults without ever blocking.
INTERACTIVE=1
if [ ! -t 0 ] || [ "$ASSUME_YES" -eq 1 ]; then INTERACTIVE=0; fi

ask() {  # ask <prompt> <default>  -> echoes the answer
  local prompt="$1" default="$2" reply=""
  if [ "$INTERACTIVE" -eq 0 ]; then echo "$default"; return; fi
  read -r -p "$prompt" reply </dev/tty || reply=""
  echo "${reply:-$default}"
}

yesno() {  # yesno <prompt> <Y|N>
  local d="$2" reply
  local hint="[Y/n]"; [ "$d" = "N" ] && hint="[y/N]"
  reply=$(ask "$1 $hint " "$d")
  case "$reply" in [Yy]*) return 0 ;; [Nn]*) return 1 ;; *) [ "$d" = "Y" ] ;; esac
}

menu() {  # print option lists only when someone is there to answer them
  if [ "$INTERACTIVE" -eq 1 ]; then printf '%s\n' "$@"; fi
  return 0
}

rule() { printf '\n%s\n' "----------------------------------------------------------------"; }

# ---------------------------------------------------------------- python ----
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "No Python found. Install Python 3.10 or newer, then run this again." >&2
  exit 1
fi

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
MAJOR=${VER%%.*}; MINOR=${VER##*.}
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
  echo "Python $VER found, but Kestrel needs 3.10 or newer." >&2
  echo "Set PYTHON=/path/to/python3.12 and run this again." >&2
  exit 1
fi
echo "Using $PY (Python $VER)"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv"
  "$PY" -m venv .venv || {
    echo "venv failed. On Debian or Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  }
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing dependencies (PySide6 is a large download the first time)"
python -m pip install -r requirements.txt

chmod +x run.sh node.sh install.sh skills/*/scripts/*.py 2>/dev/null || true

# ------------------------------------------------------------ speech ----
rule
echo "Offline speech (optional)"
echo
echo "  Piper reads replies aloud and faster-whisper transcribes dictation."
echo "  Both run locally; nothing is sent anywhere. About 150 MB."
if [ -z "$INSTALL_SPEECH" ]; then
  if yesno "  Install them?" "N"; then INSTALL_SPEECH=1; else INSTALL_SPEECH=0; fi
fi
if [ "$INSTALL_SPEECH" -eq 1 ]; then
  python -m pip install piper-tts faster-whisper sounddevice soundfile || {
    echo "  Speech engines did not install. Kestrel still works; the Speech tab" >&2
    echo "  has an Install button to retry." >&2
  }
fi

# ---------------------------------------------------------- shortcut ----
python -m kestrel.shortcut --quiet || true

# ------------------------------------------------------------- llama.cpp ----
rule
echo "llama.cpp — the inference backend Kestrel drives"
echo

# Detection is emitted as shell assignments and evaluated, rather than printed
# as delimited text. Both `read` and cmd's `for /f` collapse runs of delimiters,
# so an empty field would shift every value after it into the wrong variable.
SCAN=$(python -m kestrel.setup_backend --emit-sh 2>/dev/null || true)
HAVE_SERVER=""; HAVE_RPC=""; HAVE_VER=""; HAVE_OK=0; HAVE_MANAGED=0
HAVE_PROBLEM=""; DETECTED=cpu
if [ -n "$SCAN" ]; then eval "$SCAN"; fi

if [ -n "$HAVE_SERVER" ]; then
  echo "  Found: $HAVE_SERVER ${HAVE_VER:+($HAVE_VER)}"
  if [ -n "$HAVE_RPC" ]; then
    echo "  rpc-server:        $HAVE_RPC  — this machine can join a cluster"
  else
    echo "  rpc-server:        not present — clustering unavailable"
  fi

  if [ "$HAVE_OK" = "1" ]; then
    echo "  It runs correctly."
    DEFAULT_ACTION=2          # keep a working installation
  else
    echo
    echo "  !! This installation does not work:"
    echo "     $HAVE_PROBLEM"
    DEFAULT_ACTION=1          # a broken one should be replaced, not kept
  fi

  if [ "$HAVE_MANAGED" = "1" ]; then
    OPT1="Remove it and install a fresh copy"
  else
    OPT1="Install a fresh copy for Kestrel to use (leaves yours untouched)"
  fi

  if [ -z "$INSTALL_LLAMA" ]; then
    menu "" \
      "  What would you like to do?" \
      "    1) $OPT1" \
      "    2) Keep this installation and carry on" \
      "    3) Nothing for now — decide later in the Backend tab"
    choice=$(ask "  Choice [$DEFAULT_ACTION]: " "$DEFAULT_ACTION")
    case "$choice" in
      1) INSTALL_LLAMA=1; [ "$HAVE_MANAGED" = "1" ] && REMOVE_FIRST=1 ;;
      3) INSTALL_LLAMA=0 ;;
      *) INSTALL_LLAMA=0 ;;
    esac
  fi

  if [ "$HAVE_MANAGED" != "1" ] && [ "${INSTALL_LLAMA:-0}" -eq 1 ]; then
    echo
    echo "  Note: Kestrel did not install $HAVE_SERVER, so it will not be removed."
    echo "  The new copy goes in Kestrel's own directory and will be preferred."
    echo "  To remove the old one yourself: apt/dnf remove, brew uninstall, or"
    echo "  delete the directory it lives in."
  fi
else
  echo "  No llama.cpp installation was found on this machine."
  if [ -z "$INSTALL_LLAMA" ]; then
    if yesno "  Install it now?" "Y"; then INSTALL_LLAMA=1; else INSTALL_LLAMA=0; fi
  fi
fi

if [ "${INSTALL_LLAMA:-0}" -eq 2 ]; then
  echo
  python -m kestrel.setup_backend --uninstall || true
  INSTALL_LLAMA=0
fi

if [ "${INSTALL_LLAMA:-0}" -eq 1 ]; then

  # --- how ---
  if [ -z "$SOURCE" ]; then
    menu "" \
      "  How would you like it installed?" \
      "    1) Download an official prebuilt build   (fast, recommended)" \
      "    2) Compile from source                   (slower; needed for CUDA or ROCm on Linux)"
    choice=$(ask "  Choice [1]: " "1")
    case "$choice" in 2) SOURCE=1 ;; *) SOURCE=0 ;; esac
  fi

  # --- which accelerator ---
  if [ -z "$BACKEND" ]; then
    menu "" \
      "  Which accelerator? This machine looks like: $DETECTED" \
      "    1) auto   — detect and choose ($DETECTED)   (recommended)" \
      "    2) cpu    — no GPU acceleration" \
      "    3) cuda   — NVIDIA" \
      "    4) vulkan — any modern GPU, including Intel and older AMD" \
      "    5) hip    — AMD ROCm" \
      "    6) metal  — Apple silicon" \
      "    7) sycl   — Intel oneAPI"
    choice=$(ask "  Choice [1]: " "1")
    case "$choice" in
      2) BACKEND=cpu ;; 3) BACKEND=cuda ;; 4) BACKEND=vulkan ;;
      5) BACKEND=hip ;; 6) BACKEND=metal ;; 7) BACKEND=sycl ;;
      *) BACKEND=auto ;;
    esac
  fi

  # --- clustering ---
  if [ -z "$WITH_RPC" ]; then
    menu "" \
      "  The RPC backend lets this machine join or host a cluster, so a model" \
      "  too large for one machine can be spread across several."
    if yesno "  Include RPC support?" "Y"; then WITH_RPC=1; else WITH_RPC=0; fi
  fi

  # --- summary and go ---
  echo
  [ "$REMOVE_FIRST" -eq 1 ] && echo "  Removing the existing Kestrel-managed installation first."
  HOW=$([ "$SOURCE" -eq 1 ] && echo "source build" || echo "prebuilt release")
  RPCSTATE=$([ "$WITH_RPC" -eq 1 ] && echo on || echo off)
  echo "  Installing: $HOW, backend $BACKEND, RPC $RPCSTATE"
  echo
  ARGS=(--backend "$BACKEND")
  [ "$SOURCE" -eq 1 ] && ARGS+=(--source)
  [ "$WITH_RPC" -eq 0 ] && ARGS+=(--no-rpc)
  [ "$REMOVE_FIRST" -eq 1 ] && ARGS+=(--reinstall)
  if python -m kestrel.setup_backend "${ARGS[@]}"; then
    echo "  llama.cpp ready."
  else
    echo
    echo "  llama.cpp installation did not complete." >&2
    echo "  Kestrel is installed and usable: open it and use the Backend tab to" >&2
    echo "  retry, or point it at an existing build." >&2
  fi
else
  echo
  echo "  Skipping. Kestrel can find or install llama.cpp later from the Backend tab."
fi

rule
cat <<'DONE'

Installed.

  ./run.sh                 open the interface
  ./run.sh --cli           headless mode
  ./node.sh --mem 8192     turn this machine into a worker for the cluster
                           (it finds rpc-server the same way, and --install
                            will build it here if this machine has none)

Kestrel drives any OpenAI-compatible endpoint, so an existing llama-server,
LM Studio, llamafile or vLLM will work too. Point it at one under Settings, or
load a GGUF directly from the Models tab.
DONE
