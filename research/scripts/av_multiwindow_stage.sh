#!/bin/bash
set -euo pipefail
cd /home/jovyan/OneSearch-VL
AV_RUN="${AV_RUN:?}"
AV_INPUTS="${AV_INPUTS:?frozen input directory required}"
AV_MANIFEST=research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json
case "${1:?stage required}" in
observe)
 pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u -m research.av_train.multiwindow execute --bindings "$AV_INPUTS/bindings.private.json" --model-manifest "$AV_MANIFEST" --out-dir "$AV_RUN/observer-executed-v1"
 ;;
teacher)
 set +e
 timeout --signal=TERM --kill-after=5s "$((AV_LIMIT-100))" pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -u -m research.av_train.multiwindow_teacher --batch-dir "$AV_RUN/train21-v2" --index-dir "$AV_INPUTS/corpus-train-sft" --observer-dir "$AV_RUN/observer-executed-v1" --controller-kind onesearch --attempts 64 --seed 1021200 --temperature .7 --top-p .9 --model-tool-calls 6 --max-new-tokens 256 --out-dir "$AV_RUN/teacher-episodes-v1"
 AV_RC=$?
 set -e
 if [[ "$AV_RC" -ne 0 && "$AV_RC" -ne 124 ]]; then exit "$AV_RC"; fi
 pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -m research.av_train.recover_multiwindow_rows --episodes-dir "$AV_RUN/teacher-episodes-v1" --batch-dir "$AV_RUN/train21-v2" --out-dir "$AV_RUN/teacher-retained-v1"
 ;;
train)
 for AV_SUPERVISION in answer_only trajectory; do
  pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -u -m research.av_train.optimizer_smoke --rows "$AV_RUN/teacher-retained-v1/rows.private.json" --notes "$AV_RUN/teacher-retained-v1/notes.private.json" --dataset-quality silver_anonymous_publisher_fields --steps 32 --supervision "$AV_SUPERVISION" --out-dir "$AV_RUN/sft-$AV_SUPERVISION-v1"
 done
 ;;
evaluate)
 for AV_VARIANT in base answer_only trajectory; do
  AV_EXTRA=()
  if [[ "$AV_VARIANT" != base ]]; then AV_EXTRA=(--adapter "$AV_RUN/sft-$AV_VARIANT-v1/checkpoint"); fi
  pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -u -m research.av_train.multiwindow_teacher --mode evaluate --batch-dir research/runs/av-clue-searchqa-20261012-v1/development-context-final-v3 --index-dir "$AV_INPUTS/corpus-dev-sft" --observer-dir "$AV_RUN/observer-executed-v1" --controller-kind onesearch --attempts 16 --model-tool-calls 6 --max-new-tokens 256 --out-dir "$AV_RUN/dev-$AV_VARIANT-v1" "${AV_EXTRA[@]}"
 done
 pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -m research.av_bench.multiwindow_compare --run-dir "$AV_RUN" --out-file "$AV_RUN/paired-dev-diagnostics-v1.json"
 ;;
*) exit 2;;
esac
