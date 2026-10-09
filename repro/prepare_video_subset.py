"""Fetch one pinned official video/PNG subset via verified read-only ZIP ranges.

Only metadata rows are filtered; original PNGs and MP4 remain publisher bytes.
The official prepare.py then verifies this subset, not the full 307-case set.
"""
import io,json,shutil,urllib.request,zipfile,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent/'assets'
SOURCE=ROOT/'video-bench'
DEST=ROOT/'video-subset'
REV='8694a570d777066e053fc258623a429cc13a2294'
BASE=f'https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench/resolve/{REV}/'
info=json.loads((SOURCE/'hf-api-inventory.json').read_text())
sizes={x['rfilename']:x['size'] for x in info['siblings']}
class RangeReader(io.RawIOBase):
 def __init__(self,name):self.name=name;self.size=sizes[name];self.pos=0;self.downloaded=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if not 0<=self.pos<=self.size:raise ValueError('Invalid archive offset')
  return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
  if n<=0:return b''
  if n>32*1024*1024:raise ValueError('Refusing unbounded archive read')
  start=self.pos;end=start+n-1
  req=urllib.request.Request(BASE+self.name+f'?download=true&range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}'})
  with urllib.request.urlopen(req,timeout=90) as r:
   expected=f'bytes {start}-{end}/{self.size}'
   if r.status!=206 or r.headers.get('Content-Range')!=expected:raise RuntimeError('Unverified range response')
   data=r.read(n+1)
  if len(data)!=n:raise RuntimeError('Archive range length mismatch')
  self.pos+=n;self.downloaded+=n
  return data

def entries(name):
 reader=RangeReader(name)
 z=zipfile.ZipFile(reader)
 members={m.filename:m for m in z.infolist()}
 z.close()
 print('Indexed',name,'members',len(members),'metadata_bytes',reader.downloaded,flush=True)
 return members

def extract(item):
 archive,member=item
 target=DEST/member.filename
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.stat().st_size==member.file_size:return
 reader=RangeReader(archive)
 # Use normal ZipFile to retain standard CRC validation; index overhead is bounded.
 with zipfile.ZipFile(reader) as z:
  data=z.read(member.filename)
 if len(data)!=member.file_size:raise RuntimeError('Extracted size mismatch')
 target.write_bytes(data)
 return reader.downloaded

DEST.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in (SOURCE/'test.jsonl').read_text().splitlines()]
selected=rows[:1]
videos={x['video'] for x in selected}
media=[json.loads(x) for x in (SOURCE/'media.jsonl').read_text().splitlines()]
frames=[json.loads(x) for x in (SOURCE/'frames.jsonl').read_text().splitlines()]
media=[x for x in media if x['path'] in videos]
frames=[x for x in frames if x['video'] in videos]
for name,data in [('test.jsonl',selected),('media.jsonl',media),('frames.jsonl',frames)]:
 (DEST/name).write_text(''.join(json.dumps(x)+'\n' for x in data))
for name in ['prepare.py','extract_frames.py']:shutil.copyfile(SOURCE/name,DEST/name)
needed=videos|{f['path'] for x in frames for f in x['frames']}
tasks=[]
for archive in ['video-00001.zip','video-frames-00001.zip','video-frames-00002.zip']:
 members=entries(archive)
 for name in needed&members.keys():tasks.append((archive,members[name]));needed.remove(name)
 if not needed:break
if needed:raise RuntimeError(f'Missing members: {len(needed)}')
with ThreadPoolExecutor(max_workers=8) as pool: counts=list(pool.map(extract,tasks))
receipt={'revision':REV,'subset_ids':[x['id'] for x in selected],'samples':len(selected),'videos':len(videos),'files':len(tasks),'range_download_bytes':sum(x or 0 for x in counts),'protocol':'filtered original metadata; original MP4/PNG bytes; ZIP CRC checked; full dataset not evaluated'}
(DEST/'subset-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
