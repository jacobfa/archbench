#!/usr/bin/env bash
# usage: ./run.sh <run_name> <config.yaml> [more.yaml ...] [key=value ...]
#   NGPUS=4 ./run.sh fftnet-65m configs/65m/fftnet.yaml train.steps=2000
#   HOSTFILE=hosts ./run.sh gdn-380m configs/380m/gdn.yaml configs/optim_adamw_zero2.yaml
#   RESUME=1 ./run.sh fftnet-65m configs/65m/fftnet.yaml
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_name> <config.yaml> [overrides...]" >&2
  exit 1
fi

NAME=$1; shift
LOG_DIR=${LOG_DIR:-logs}
RUN_LOG_DIR="$LOG_DIR/$NAME"
mkdir -p "$RUN_LOG_DIR"
LOG_FILE="$RUN_LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

ARGS=("$@" "wandb.name=$NAME" "train.log_dir=$LOG_DIR")
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=("train.resume=$NAME")
fi

if [[ -n "${HOSTFILE:-}" ]]; then
  LAUNCH=(deepspeed --hostfile "$HOSTFILE")
else
  NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  LAUNCH=(deepspeed --num_gpus "${NGPUS:-1}")
fi

echo "run:  $NAME"
echo "log:  $LOG_FILE"
echo "cmd:  ${LAUNCH[*]} train.py ${ARGS[*]}"
"${LAUNCH[@]}" train.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
