#!/usr/bin/env bash
# Sourced by inference/evaluation wrappers before any Python imports.
# Never trace service credentials, including when invoked with bash -x.
set +x
INFER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ENV_FILE:-}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
elif [[ -f "${INFER_ENV_DIR}/.env" ]]; then
    set -a
    source "${INFER_ENV_DIR}/.env"
    set +a
fi
