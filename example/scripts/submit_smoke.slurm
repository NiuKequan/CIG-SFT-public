#!/usr/bin/env bash
# GPU smoke test: run the 1.5B recipe for a few dozen steps on one card.
#
# It proves what a CPU run cannot: the two views and the reference model are dispatched to the right device, the
# reference model is loaded once and left frozen, and the step does not run out of memory.
#
# Usage:
#   sbatch example/scripts/submit_smoke.slurm
#   sbatch --gres=gpu:a100-pcie-40gb:1 example/scripts/submit_smoke.slurm
#   SMOKE_STEPS=100 sbatch --export=ALL,SMOKE_STEPS=100 example/scripts/submit_smoke.slurm
#
# Adjust the partition, account and gres lines to your cluster. Cancel as soon as the log shows the loss going down:
#   scancel <jobid>

#SBATCH -J cig-sft-smoke
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -o cig-sft-smoke-%j.out
#SBATCH -e cig-sft-smoke-%j.err

set -euo pipefail

# `sbatch` copies this file into its spool directory before running it, so `BASH_SOURCE` points there and not at the
# repository. Submit from the repository root, or pass `ROOT=/path/to/CIG-SFT` explicitly.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

SMOKE_STEPS="${SMOKE_STEPS:-30}"
# The credit is read once over the whole training set before the loop starts, so a full math-30k smoke run spends most
# of its hour there. Set `SMOKE_SAMPLES` to shrink both phases, e.g. `SMOKE_SAMPLES=1024 SMOKE_STEPS=8`.
SMOKE_SAMPLES="${SMOKE_SAMPLES:-}"
OUT="${OUT:-$ROOT/saves/smoke_gpu}"

[[ -f "$ROOT/example/scripts/common.sh" ]] || {
  echo "cannot find the repository at $ROOT, submit from its root or pass ROOT=/path/to/CIG-SFT" >&2
  exit 2
}

source "$ROOT/example/scripts/common.sh"

CONFIG="$EXAMPLE/configs/qwen2_5_math_1p5b_full_cig_sft.yaml"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B}"

# the compute nodes have no outbound network, and the hugging face cache usually lives on a node local tmpdir, so fetch
# the weights on the login node, into a path that the compute node can see. The datasets already ship in example/data:
#   huggingface-cli download Qwen/Qwen2.5-Math-1.5B    # fills ~/.cache/huggingface for the compute node
#   python example/scripts/build_dataset.py --task numina_cot --sizes 30k   # only to rebuild the parquet
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ -f "$CONFIG" ]] || die "missing $CONFIG"
[[ -f "$EXAMPLE/data/numina_cot/30k/train.parquet" ]] || die "missing the numina-cot-30k parquet, see the comment above"

check_python_env
log "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'no nvidia-smi')"
log "running $SMOKE_STEPS steps from $MODEL, output in $OUT (charged per requested hour, cancel when the loss is finite)"

# Overrides are `key=value` and not `--key value`: with a YAML config the launcher merges them through
# `OmegaConf.from_cli`, which only understands the former. A bare `no` is read as the YAML 1.1 boolean and then fails
# HfArgumentParser's enum check, so the string valued override has to be quoted inside the token.
# `save_strategy=no` matters: the trainer writes a final checkpoint even when `max_steps` cuts the run short of
# `save_steps`, and that checkpoint is a full copy of the 1.5B weights.
# `max_samples` is only added when it is set, so the default keeps the whole training set
SAMPLES_ARGS=()
if [[ -n "$SMOKE_SAMPLES" ]]; then
  SAMPLES_ARGS=("max_samples=$SMOKE_SAMPLES")
  log "limiting the run to $SMOKE_SAMPLES examples"
fi

run_train "$CONFIG" \
  "model_path=$MODEL" \
  "dataset_dir=$EXAMPLE/data" \
  "output_dir=$OUT" \
  "cig_credit_cache=$OUT/credit.pt" \
  "cutoff_len=1024" \
  "max_steps=$SMOKE_STEPS" \
  'save_strategy="no"' \
  "plot_loss=false" \
  "${SAMPLES_ARGS[@]}" \
  "$@"

log "smoke run finished, the credit cache is at $OUT/credit.pt"
