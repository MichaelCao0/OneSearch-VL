#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  echo "Usage: $0 {image|video} [evaluation arguments]"
}
case "${1:-}" in
  image|video) MODALITY="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"
exec python "${SCRIPT_DIR}/eval_${MODALITY}.py" "$@"
