#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"
exec python "${SCRIPT_DIR}/run_infer.py" "$@"
