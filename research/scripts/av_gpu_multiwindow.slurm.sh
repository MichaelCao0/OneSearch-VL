#!/bin/bash
#SBATCH --job-name=onesearch-av-multiwindow
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-av-multiwindow-%j.out
#SBATCH --error=/tmp/onesearch-av-multiwindow-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
AV_RUN=research/runs/av-multiwindow-20261012-v1
export AV_RUN
AV_INPUTS="$AV_RUN/gpu-inputs-v2"
export AV_INPUTS
AV_STAGE="${1:?observe, teacher, train, or evaluate required}"
case "$AV_STAGE" in observe) AV_LIMIT=900;; teacher) AV_LIMIT=1200;; train) AV_LIMIT=600;; evaluate) AV_LIMIT=900;; *) exit 2;; esac
export AV_STAGE AV_LIMIT
AV_CLUSTER="$AV_RUN/cluster-$AV_STAGE-v1"
export AV_CLUSTER
mkdir "$AV_CLUSTER"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python - <<'PY'
import json,os,subprocess
from pathlib import Path
from research.av_data.media_clock import sha256_file
p=json.loads((Path(os.environ['AV_INPUTS'])/'preflight.private.json').read_text())
assert p['git_head']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for path,value in p['locks'].items():assert sha256_file(path)==value
subprocess.run(['git','diff','--quiet'],check=True)
subprocess.run(['git','diff','--cached','--quiet'],check=True)
PY
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_CLUSTER/driver-probe.json" --mask-output "$AV_CLUSTER/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_CLUSTER/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python - <<'PY'
from pathlib import Path
import json,os,torch
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
root=Path(os.environ['AV_CLUSTER']);assigned=json.loads((root/'driver-probe.json').read_text());visible=driver_inventory()
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and len(visible)==1
assert bus_key(visible[0]['pci_bus_id'])==bus_key(assigned['assigned_pci_bus_id'])
a=torch.ones((32,32),device='cuda:0');b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
(root/'gpu-runtime.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'allocation_pci_matched':True,'actual_matmul_passed':True}))
PY
exec timeout --signal=TERM --kill-after=5s "$((AV_LIMIT-35))" bash research/scripts/av_multiwindow_stage.sh "$AV_STAGE"
