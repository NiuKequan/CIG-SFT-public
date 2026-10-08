#!/usr/bin/env bash
# Install the pinned dependencies and this repository into the currently active Python environment.
# Create and activate an environment separately (conda, venv, uv, or another environment manager) before running this script.
#
# Usage:
#   bash example/scripts/setup_env.sh
#
# flash-attn is not part of the pinned environment. Install it with:
#   pip install flash-attn --no-build-isolation

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check_python_env
log "installing the pinned dependencies from example/requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$EXAMPLE/requirements.txt"

# `--no-deps` is safe: every dependency of `pyproject.toml` is already pinned by the file above.
log "installing llamafactory in editable mode"
python -m pip install --no-deps -e "$ROOT"

log "checking the dependency versions"
python -c "from llamafactory.extras.misc import check_dependencies; check_dependencies()"

log "installed version"
python -c "from llamafactory.extras.env import VERSION; print(VERSION)"

log "done; the packages were installed into the currently active Python environment"
