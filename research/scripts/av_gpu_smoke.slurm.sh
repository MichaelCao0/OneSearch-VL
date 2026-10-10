#!/bin/bash
#SBATCH --job-name=onesearch-av-native-smoke
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/home/jovyan/OneSearch-VL/research/runs/av-native-smoke-20261010-v1/cluster/smoke-%j.out
#SBATCH --error=/home/jovyan/OneSearch-VL/research/runs/av-native-smoke-20261010-v1/cluster/smoke-%j.err
set -euo pipefail
umask 077
cd /home/jovyan/OneSearch-VL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
if [[ ${NVIDIA_VISIBLE_DEVICES:-} =~ ^(GPU-[0-9a-fA-F-]+|MIG-[0-9a-fA-F-]+|[0-9]+)$ ]]; then
  export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
fi
exec timeout --signal=TERM --kill-after=15s 1720 bash -c '
set -euo pipefail
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python - <<"PY"
from pathlib import Path
import json,os,datetime,torch
p=Path("research/runs/av-native-smoke-20261010-v1/cluster/gpu-runtime.json")
if p.exists():raise RuntimeError("refuse overwrite")
n=torch.cuda.device_count()
d={"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"visible_device_count":n,"NVIDIA_VISIBLE_DEVICES":os.getenv("NVIDIA_VISIBLE_DEVICES"),"CUDA_VISIBLE_DEVICES":os.getenv("CUDA_VISIBLE_DEVICES"),"device_names":[torch.cuda.get_device_name(i) for i in range(n)],"allocation_isolation_valid":n==1}
p.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
assert n==1,"Refuse inference without single allocated GPU isolation"
a=torch.ones((32,32),device="cuda:0");b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
print("actual_cuda_matmul_passed",flush=True)
PY
pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -u research/scripts/run_av_observer_smoke.py --bindings research/runs/av-native-smoke-20261010-v1/bindings/inputs.private.json --model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json --output-dir research/runs/av-native-smoke-20261010-v1/inference-v1 --device cuda:0 --dtype bfloat16 --max-clips 6 --max-clip-seconds 2 --max-new-tokens 384 --video-fps 2 --max-pixels 200704
'
