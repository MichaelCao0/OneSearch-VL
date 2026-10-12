#!/bin/bash
#SBATCH --job-name=onesearch-native-omni-search
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-native-omni-%j.out
#SBATCH --error=/tmp/onesearch-native-omni-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
AV_NATIVE_RUN="${1:?versioned run directory required}"
AV_NATIVE_BATCH="${2:?batch directory required}"
AV_NATIVE_TASKS="${3:-16}"
[[ "$AV_NATIVE_RUN" =~ ^research/runs/av-native-searchqa-[0-9]{8}-v[0-9]+/gpu-v[0-9]+$ ]]
export AV_NATIVE_RUN
mkdir -p "$AV_NATIVE_RUN/cluster"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_NATIVE_RUN/cluster/driver-probe.json" --mask-output "$AV_NATIVE_RUN/cluster/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_NATIVE_RUN/cluster/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python - <<'PY'
from pathlib import Path
import json,os,torch
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
root=Path(os.environ['AV_NATIVE_RUN']);assigned=json.loads((root/'cluster/driver-probe.json').read_text());visible=driver_inventory()
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and len(visible)==1
assert bus_key(visible[0]['pci_bus_id'])==bus_key(assigned['assigned_pci_bus_id'])
a=torch.ones((32,32),device='cuda:0');b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
(root/'cluster/gpu-runtime.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'allocation_pci_matched':True,'actual_matmul_passed':True}))
PY
exec timeout --signal=TERM --kill-after=10s 1420 pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u -m research.av_train.native_searchqa \
 --batch-dir "$AV_NATIVE_BATCH" --index-dir "$AV_NATIVE_RUN/corpus" --model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json \
 --out-dir "$AV_NATIVE_RUN/episodes" --max-tasks "$AV_NATIVE_TASKS" --max-new-tokens 512 --max-turns 6 --prompt-file onesearch_vl/onesearch_infer/prompts/audio_video_search_v2.txt
