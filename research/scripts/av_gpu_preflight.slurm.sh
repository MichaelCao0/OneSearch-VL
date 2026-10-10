#!/bin/bash
#SBATCH --job-name=onesearch-av-preflight
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gi
#SBATCH --output=/home/jovyan/OneSearch-VL/research/runs/av-native-smoke-20261010-v1/cluster/preflight-%j.out
#SBATCH --error=/home/jovyan/OneSearch-VL/research/runs/av-native-smoke-20261010-v1/cluster/preflight-%j.err
set -euo pipefail
cd /home/jovyan/OneSearch-VL
mkdir -p research/runs/av-native-smoke-20261010-v1/cluster
exec timeout --signal=TERM --kill-after=15s 240 bash -c '
set -euo pipefail
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
command -v pixi
test -r AGENTS.md
test -r research/runs/next-stage-20261010-v1/cases-v3/sources.jsonl
pixi run python - <<"PY"
from pathlib import Path
import hashlib,json,datetime,os
p=Path("research/runs/av-native-smoke-20261010-v1/cluster/preflight.json")
if p.exists():raise RuntimeError("refuse overwrite")
f=Path("research/runs/next-stage-20261010-v1/cases-v3/sources.jsonl")
d={"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"workspace":"/home/jovyan/OneSearch-VL","sources_registry_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"shared_home_readable":True,"shared_home_write_probe":True,"job_index":os.environ.get("JOB_COMPLETION_INDEX"),"gpu_runtime_pending":True}
p.write_text(json.dumps(d,indent=2));print(json.dumps(d))
PY
'
