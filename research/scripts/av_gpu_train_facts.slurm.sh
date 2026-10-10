#!/bin/bash
#SBATCH --job-name=onesearch-av-train-facts
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/tmp/onesearch-av-train-facts-%j.out
#SBATCH --error=/tmp/onesearch-av-train-facts-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
export AV_FACT_RUN="${1:?run directory required}"
[[ "$AV_FACT_RUN" =~ ^research/runs/av-data-pipeline-[0-9]{8}-v[0-9]+/fact-verification-v[0-9]+$ ]]
export AV_FACT_CLUSTER="$AV_FACT_RUN/cluster"
test -f "$AV_FACT_RUN/bindings.private.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec timeout --signal=TERM --kill-after=15s 620 bash -c '
set -euo pipefail
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python research/scripts/av_gpu_driver_probe.py --output "$AV_FACT_CLUSTER/driver-probe.json" --mask-output "$AV_FACT_CLUSTER/cuda-mask.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS= read -r CUDA_VISIBLE_DEVICES < "$AV_FACT_CLUSTER/cuda-mask.txt"
export CUDA_VISIBLE_DEVICES
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python - <<"PY"
from pathlib import Path
import json,os,datetime,torch
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
p=Path(os.environ["AV_FACT_CLUSTER"])/"gpu-runtime.json"
assert not p.exists(),"refuse overwrite"
assigned=json.loads((p.parent/"driver-probe.json").read_text())
assert torch.cuda.is_available() and torch.cuda.device_count()==1,"require one usable allocated GPU"
visible=driver_inventory()
assert len(visible)==1 and bus_key(visible[0]["pci_bus_id"])==bus_key(assigned["assigned_pci_bus_id"]),"require exact assigned PCI"
a=torch.ones((32,32),device="cuda:0");b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
r={"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"device":torch.cuda.get_device_name(0),"visible_device_count":1,"allocation_pci_matched":True,"actual_matmul_passed":True}
p.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
PY
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u research/scripts/run_av_observer_smoke.py --bindings "$AV_FACT_RUN/bindings.private.json" --model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json --output-dir "$AV_FACT_RUN/model-observations-v1" --clock-policy av_pts_grid_v1 --output-policy json_object_v1 --device cuda:0 --dtype bfloat16 --max-clips 6 --max-clip-seconds 2 --max-new-tokens 384 --video-fps 2 --max-pixels 200704
'
