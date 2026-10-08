#!/usr/bin/env bash
# Train the recipe the paper reports: Qwen2.5-Math-1.5B on math-30k, 1 epoch, 233 steps, global batch 128.
#
# Usage:
#   bash example/scripts/train_math_30k.sh
#   bash example/scripts/train_math_30k.sh --num_train_epochs 1 --max_steps 20   # any config key can be overridden
#
# Prepare the environment and the data first:
#   bash example/scripts/setup_env.sh
#   python example/scripts/build_dataset.py --task numina_cot --sizes 30k
#
# On a cluster use example/scripts/submit_smoke.slurm as the template, one 40GB card is enough for the 1.5B model.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG="$EXAMPLE/configs/qwen2_5_math_1p5b_full_cig_sft.yaml"
PARQUET="$EXAMPLE/data/numina_cot/30k/train.parquet"

[[ -f "$CONFIG" ]] || die "missing $CONFIG"
[[ -f "$PARQUET" ]] || die "missing $PARQUET, run:
  python example/scripts/build_dataset.py --task numina_cot --sizes 30k"

check_python_env
log "training from $CONFIG"
log "the credit is read once and cached, watch for 'Read the CIG-SFT credit of N new ...'"

run_train "$CONFIG" "$@"
