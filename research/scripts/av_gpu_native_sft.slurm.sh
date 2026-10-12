#!/bin/bash
#SBATCH --job-name=onesearch-native-omni-sft
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-native-sft-%j.out
#SBATCH --error=/tmp/onesearch-native-sft-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
AV_SFT_RUN="${1:?versioned native SFT run directory required}"
[[ "$AV_SFT_RUN" =~ ^research/runs/av-native-sft-[0-9]{8}-v[0-9]+/gpu-v[0-9]+$ ]]
export AV_SFT_RUN
mkdir -p "$AV_SFT_RUN/cluster"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_SFT_RUN/cluster/driver-probe.json" --mask-output "$AV_SFT_RUN/cluster/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_SFT_RUN/cluster/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python research/scripts/av_native_sft_preflight.py
export PYTHONPATH=SFT/src
for AV_SFT_VARIANT in trajectory answer_only; do
  timeout --signal=TERM --kill-after=10s 850 pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -u SFT/src/train.py "$AV_SFT_RUN/train-$AV_SFT_VARIANT.yaml" > "$AV_SFT_RUN/train-$AV_SFT_VARIANT.log" 2>&1
 done
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python "$AV_SFT_RUN/validate_checkpoints.py"
