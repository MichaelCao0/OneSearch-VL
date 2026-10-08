#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
CONFIG="examples/onesearch/qwen3_vl_8b_joint_110k.yaml"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' llamafactory-cli train "$CONFIG" "$@"
  printf '\n'
  exit 0
fi
exec llamafactory-cli train "$CONFIG" "$@"
