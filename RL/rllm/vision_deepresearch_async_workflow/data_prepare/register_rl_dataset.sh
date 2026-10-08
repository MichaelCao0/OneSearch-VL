#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../../../../../OneSearch-VL-RL-10K}"
python "${SCRIPT_DIR}/register_rl_dataset.py" \
  --name "${RL_DATASET_NAME:-OneSearch-VL-RL-10K}" \
  --train "${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}" "$@"
