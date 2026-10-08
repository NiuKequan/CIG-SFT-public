#!/usr/bin/env bash
# Prepare the released math-30k data if needed, then launch the published recipe.
# The model is pulled by Transformers from the model_path in the recipe
# (override it with model_path=/path/to/model).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check_python_env
if [[ ! -f "$EXAMPLE/data/numina_cot/30k/train.parquet" ]]; then
  python "$EXAMPLE/scripts/build_dataset.py" --task numina_cot --sizes 30k
fi

exec bash "$EXAMPLE/scripts/train_math_30k.sh" "$@"
