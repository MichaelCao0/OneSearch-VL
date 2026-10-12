"""Verify allocated GPU and frozen native trainer/model artifacts before SFT."""
from pathlib import Path
import hashlib,json,os,sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.scripts.av_gpu_driver_probe import driver_inventory,bus_key
root=Path(os.environ['AV_SFT_RUN']);assigned=json.loads((root/'cluster/driver-probe.json').read_text());visible=driver_inventory()
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and len(visible)==1
assert bus_key(visible[0]['pci_bus_id'])==bus_key(assigned['assigned_pci_bus_id'])
a=torch.ones((32,32),device='cuda:0');b=a@a;torch.cuda.synchronize();assert b[0,0].item()==32
freeze=json.loads((root/'cluster/source-freeze.json').read_text())
for file,expected in freeze['files'].items():
 assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected, 'source changed: '+file
manifest=json.loads(Path('research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json').read_text())
assert manifest['status']=='verified' and manifest['all_lfs_verified']
verified_bytes=0
for record in manifest['files']:
 h=hashlib.sha256();artifact=Path(record['local_path'])
 with artifact.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==record['expected_sha256'], 'model artifact changed: '+record['path']
 verified_bytes+=artifact.stat().st_size
(root/'cluster/model-verification.json').write_text(json.dumps({'all_artifacts_rehashed':True,'verified_bytes':verified_bytes,'revision':manifest['revision']}))
(root/'cluster/gpu-runtime.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'allocation_pci_matched':True,'actual_matmul_passed':True,'native_trainer':'SFT/src/train.py'}))
