"""Reproducible sampling priorities; strata are suggestions, never content gold."""
import re
from collections import defaultdict,deque
from research.av_train.source_isolation import canonical_url
CYCLE=('music','sports','live','music','sports','other','music','sports','live','other')
SPORTS=re.compile(r'\b(hockey|football|basketball|tennis|cricket|baseball|rugby|volleyball|badminton|skating|skiing|ski|curling|wrestling|fencing|archery|rowing|marathon|cycling|surfing|skateboard|racing|race|competition|overtime|tournament)\b',re.I)
LIVE=re.compile(r'\b(festival|parade|carnival|procession|ceremony|concert|performance|show|stunt|dance|demonstration|rehearsal|band|choir|opera|orchestra)\b',re.I)
def candidate_stratum(row):
 domain=row.get('candidate_domain','').split('_')[0]
 text=row.get('query','')+' '+str(row.get('public_metadata',{}).get('title',''))
 if domain=='music':return 'music'
 if domain=='sports' or SPORTS.search(text):return 'sports'
 if domain=='live' or LIVE.search(text):return 'live'
 return 'other'
def schedule_sources(rows):
 queues=defaultdict(deque)
 for row in sorted(rows,key=lambda r:r['source_id']):queues[candidate_stratum(row)].append(row)
 out=[]
 while any(queues.values()):
  for key in CYCLE:
   if queues[key]:out.append(queues[key].popleft())
 return out

def reserve_candidates(rows,families,*,limit,blocked_source_ids=(),blocked_urls=(),blocked_hashes=()):
 if not isinstance(limit,int) or not 1<=limit<=1000:raise ValueError('Explicit recording cap required')
 groups={}
 for group in families:
  for sid in group['source_ids']:
   if sid in groups:raise ValueError('Ambiguous family')
   groups[sid]=group['candidate_family_id']
 if any(r['source_id'] not in groups for r in rows):raise ValueError('Missing source family')
 blocked={groups.get(s,s) for s in blocked_source_ids};urls={canonical_url(u) for u in blocked_urls};hashes=set(blocked_hashes)
 for row in rows:
  if canonical_url(row['origin_url']) in urls or row['sha256'] in hashes:blocked.add(groups[row['source_id']])
 out=[]
 for row in schedule_sources(rows):
  family=groups[row['source_id']]
  if family in blocked or row.get('status')!='candidate_unreviewed_av' or not row.get('description_page_html_path'):continue
  out.append(row);blocked.add(family)
  if len(out)>=limit:break
 return out
