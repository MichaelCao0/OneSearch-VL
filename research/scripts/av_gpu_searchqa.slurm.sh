#!/bin/bash
#SBATCH --job-name=onesearch-av-searchqa
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-av-searchqa-%j.out
#SBATCH --error=/tmp/onesearch-av-searchqa-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
export AV_QA_RUN="${1:?run directory required}"
[[ "$AV_QA_RUN" =~ ^research/runs/av-searchqa-[0-9]{8}-v[0-9]+/(bootstrap-v[0-9]+|guided-seed-v[0-9]+)$ ]]
export AV_QA_CLUSTER="$AV_QA_RUN/cluster"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec timeout --signal=TERM --kill-after=15s 200 bash -c '
set -euo pipefail
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_QA_CLUSTER/driver-probe.json" --mask-output "$AV_QA_CLUSTER/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_QA_CLUSTER/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u - <<"PY"
from pathlib import Path
import json,os,torch
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
root=Path(os.environ["AV_QA_CLUSTER"])
assigned=json.loads((root/"driver-probe.json").read_text())
assert torch.cuda.is_available() and torch.cuda.device_count()==1
visible=driver_inventory()
assert len(visible)==1 and bus_key(visible[0]["pci_bus_id"])==bus_key(assigned["assigned_pci_bus_id"])
a=torch.ones((32,32),device="cuda:0");b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
result={"device":torch.cuda.get_device_name(0),"torch":torch.__version__,"allocation_pci_matched":True,"actual_matmul_passed":True}
(root/"gpu-runtime.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u -m research.av_train.searchqa \
 --batch-dir research/runs/av-searchqa-20261011-v1/batch \
 --index-dir research/runs/av-searchqa-20261011-v1/corpus-gpu-runtime-v1 \
 --checkpoint repro/assets/model-8b \
 --controller-kind omni --controller-model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json \
 --observer-smoke-dir research/runs/av-data-pipeline-20261010-v2/fact-verification-v2/model-observations-v1 \
 --out-dir "$AV_QA_RUN/episodes" --guided-curriculum --max-tasks 3 --max-new-tokens 384 \
 --task-id onesearch-av-train-001 --task-id onesearch-av-train-005 --task-id onesearch-av-train-009
'
