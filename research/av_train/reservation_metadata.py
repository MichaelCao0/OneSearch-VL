"""Join publisher metadata to sparse source reservations without changing identity."""
def enrich_reservations(reservations,seeds):
 by_id={s['source_id']:s for s in seeds}
 if len(by_id)!=len(seeds):raise ValueError('Ambiguous source identity')
 out=[]
 for row in reservations:
  result=dict(row)
  if 'public_metadata' not in result:
   seed=by_id.get(row['source_id'])
   if seed is None or seed.get('sha256')!=row['asset_sha256'] or 'public_metadata' not in seed:raise ValueError('Reservation metadata requires exact source bytes')
   result['public_metadata']=dict(seed['public_metadata'])
  out.append(result)
 return out
