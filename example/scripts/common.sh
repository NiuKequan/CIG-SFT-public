#!/usr/bin/env bash
# Shared helpers of the CIG-SFT example scripts. Source this file, do not run it.

set -euo pipefail

# EXAMPLE is this folder (`example/`), ROOT is the repository root that holds the LlamaFactory source tree.
EXAMPLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$EXAMPLE/.." && pwd)"
# 3.11 is the floor: the trainer uses `enum.StrEnum`.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

check_python_env() {
  command -v python >/dev/null 2>&1 || die "python was not found; activate your training environment first"
  python - "$PYTHON_VERSION" <<'PY_CHECK'
import sys
required = tuple(map(int, sys.argv[1].split(".")))
if sys.version_info[:2] < required:
    raise SystemExit(f"Python {required[0]}.{required[1]} or newer is required, found {sys.version.split()[0]}")
PY_CHECK
  log "using the current Python environment ($(python -V 2>&1))"
}

# Launch `src/train.py` with plain `python` on a single device and with `torchrun` when several GPUs are visible,
# which is the local form the README documents. Set NPROC_PER_NODE to override the process count.
run_train() {
  local ngpu="${NPROC_PER_NODE:-}"
  if [[ -z "$ngpu" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      ngpu="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
      ngpu=1
    fi
  fi

  [[ "$ngpu" -gt 0 ]] || ngpu=1
  if [[ "$ngpu" -gt 1 ]]; then
    log "launching torchrun on $ngpu devices"
    torchrun --standalone --nnodes=1 --nproc_per_node="$ngpu" "$ROOT/src/train.py" "$@"
  else
    python "$ROOT/src/train.py" "$@"
  fi
}
