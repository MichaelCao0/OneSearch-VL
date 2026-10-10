#!/bin/bash
#SBATCH --job-name=onesearch-av-observer-diagnostic
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-av-observer-%j.out
#SBATCH --error=/tmp/onesearch-av-observer-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
AV_DIAGNOSTIC_LABEL="${1:?diagnostic label required}"
AV_CLOCK_POLICY="${2:?clock policy required}"
[[ "$AV_DIAGNOSTIC_LABEL" =~ ^(protocol|clock|avgrid)-v[1-9][0-9]*$ ]]
[[ "$AV_CLOCK_POLICY" == strict_origin || "$AV_CLOCK_POLICY" == audio_pts_v1 || "$AV_CLOCK_POLICY" == av_pts_grid_v1 ]]
AV_OUTPUT_POLICY="${3:-unconstrained}"
[[ "$AV_OUTPUT_POLICY" == unconstrained || "$AV_OUTPUT_POLICY" == json_object_v1 ]]
export AV_DIAGNOSTIC_LABEL AV_CLOCK_POLICY AV_OUTPUT_POLICY
export AV_RUN_ROOT=research/runs/av-observer-repair-20261010-v1
export AV_DIAGNOSTIC_CLUSTER="$AV_RUN_ROOT/cluster/$AV_DIAGNOSTIC_LABEL"
mkdir -p "$AV_DIAGNOSTIC_CLUSTER"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec timeout --signal=TERM --kill-after=15s 620 bash -c '
set -euo pipefail
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_DIAGNOSTIC_CLUSTER/driver-probe.json" --mask-output "$AV_DIAGNOSTIC_CLUSTER/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_DIAGNOSTIC_CLUSTER/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python - <<"PY"
from pathlib import Path
import json,os,datetime,torch
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
p=Path(os.environ["AV_DIAGNOSTIC_CLUSTER"])/"gpu-runtime.json"
assert not p.exists(),"refuse overwrite"
assigned=json.loads((p.parent/"driver-probe.json").read_text())
assert torch.cuda.is_available() and torch.cuda.device_count()==1,"require one usable allocated GPU"
visible=driver_inventory()
assert len(visible)==1 and bus_key(visible[0]["pci_bus_id"])==bus_key(assigned["assigned_pci_bus_id"]),"require exact assigned PCI"
a=torch.ones((32,32),device="cuda:0");b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
r={"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"device":torch.cuda.get_device_name(0),"visible_device_count":1,"allocation_pci_matched":True,"actual_matmul_passed":True,"clock_policy":os.environ["AV_CLOCK_POLICY"]}
p.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
PY
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u research/scripts/run_av_observer_smoke.py --bindings research/runs/av-native-smoke-20261010-v1/bindings/inputs.private.json --model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json --output-dir "$AV_RUN_ROOT/$AV_DIAGNOSTIC_LABEL" --clock-policy "$AV_CLOCK_POLICY" --output-policy "$AV_OUTPUT_POLICY" --device cuda:0 --dtype bfloat16 --max-clips 6 --max-clip-seconds 2 --max-new-tokens 384 --video-fps 2 --max-pixels 200704
'
