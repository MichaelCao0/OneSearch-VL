"""Remove frozen held-out pages and recording candidates before teacher rollout."""
from urllib.parse import unquote
def canonical_url(value):
 return unquote(value).replace(' ','_').rstrip('/')
def partition_documents(documents,blocked_urls=(),blocked_hashes=(),blocked_source_ids=()):
 urls={canonical_url(u) for u in blocked_urls};hashes=set(blocked_hashes);ids=set(blocked_source_ids)
 kept=[];removed=[]
 for row in documents:
  reasons=[]
  if canonical_url(row['origin_url']) in urls:reasons.append('reserved_public_url')
  if row.get('asset_sha256') in hashes:reasons.append('reserved_media_hash')
  if row['source_id'] in ids:reasons.append('reserved_family_source')
  if reasons:removed.append({'source_id':row['source_id'],'origin_url':row['origin_url'],'reasons':reasons})
  else:kept.append(row)
 return kept,removed
