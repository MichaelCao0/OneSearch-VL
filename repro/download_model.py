"""Pinned public weight download using bounded ranges and mandatory LFS hashes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,time,urllib.request
ROOT=Path(__file__).resolve().parent/'assets'/'model-8b'
SCRATCH=Path('/tmp/onesearch-vl-model-2623603767b9')
REV='2623603767b9fef7bab05ad0c3cd4d464301e8df'
SIZE=17534340584
EXPECTED='f73186b544f7ff267d1150cc5ef3340edbf196d21b46794e4a623734304c8c5b'
URL=f'https://huggingface.co/OneSearch-VL/OneSearch-VL-8B/resolve/{REV}/model.safetensors'
WIDTH=64*1024*1024
SCRATCH.mkdir(parents=True,exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 return h.hexdigest()
def get_part(item):
 i,s,e=item;length=e-s+1;part=SCRATCH/f'chunk-{i:04d}'
 if part.exists() and part.stat().st_size==length:return part
 for attempt in range(3):
  try:
   req=urllib.request.Request(URL+f'?download=true&bounded_range={s}-{e}&attempt={attempt}',headers={'Range':f'bytes={s}-{e}'})
   with urllib.request.urlopen(req,timeout=90) as r:
    expected=f'bytes {s}-{e}/{SIZE}'
    if r.status!=206 or r.headers.get('Content-Range')!=expected:raise RuntimeError('Invalid range response')
    with part.open('wb') as f:
     while b:=r.read(8*1024*1024):f.write(b)
     f.flush();os.fsync(f.fileno())
   if part.stat().st_size!=length:raise RuntimeError('Incomplete range')
   if i%16==0:print(f'Complete bounded chunk {i}: [{s},{e}]',flush=True)
   return part
  except Exception as exc:
   print('Retry',i,attempt,type(exc).__name__,flush=True)
   if attempt==2:raise
   time.sleep(2)
ranges=[(i,s,min(SIZE-1,s+WIDTH-1)) for i,s in enumerate(range(0,SIZE,WIDTH))]
with ThreadPoolExecutor(max_workers=8) as ex:parts=list(ex.map(get_part,ranges))
assembled=SCRATCH/'model.safetensors'
h=hashlib.sha256()
with assembled.open('wb') as out:
 for p in parts:
  with p.open('rb') as f:
   while b:=f.read(8*1024*1024):out.write(b);h.update(b)
 out.flush();os.fsync(out.fileno())
if assembled.stat().st_size!=SIZE or h.hexdigest()!=EXPECTED:raise RuntimeError(f'Source integrity failed: {h.hexdigest()}')
print('Scratch assembly LFS hash verified',flush=True)
final=ROOT/'model.safetensors'
copy=ROOT/'model.verified-copy'
shutil.copyfile(assembled,copy)
if copy.stat().st_size!=SIZE or digest(copy)!=EXPECTED:raise RuntimeError('Persistent readback integrity failed')
copy.rename(final)
receipt={'repo':'OneSearch-VL/OneSearch-VL-8B','revision':REV,'bytes':SIZE,'sha256':EXPECTED,'expected_lfs_sha256':EXPECTED,'verified':True,'protocol':'64MiB strict HTTP ranges to local scratch; source hash and persistent readback hash'}
(ROOT/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
