"""Task outcomes from frozen predictions and silver ratings, never tool counts."""
def outcome(prediction, rating):
 if prediction is None or prediction.get('status')!='ok' or rating is None:
  return {'answer_success':None,'supported_success':None,'failure_category':'unassessed'}
 for key in ['all_fields_correct','contradicts_reference','unsupported_assertion']:
  if type(rating.get(key)) is not bool:raise ValueError('Typed semantic rating required')
 complete=prediction.get('protocol_final') is True
 correct=complete and rating['all_fields_correct'] and not rating['contradicts_reference']
 supported=correct and not rating['unsupported_assertion']
 category='success' if supported else 'unsupported_extra_claim' if correct else 'no_final' if not complete else 'wrong_answer'
 return {'answer_success':correct,'supported_success':supported,'failure_category':category}

def summarize_outcomes(predictions, ratings, planned_ids):
 if len(planned_ids)!=len(set(planned_ids)):raise ValueError('Distinct planned questions required')
 if set(predictions)-set(planned_ids) or set(ratings)-set(planned_ids):raise ValueError('Unknown task in outcome inputs')
 rows=[outcome(predictions.get(tid),ratings.get(tid)) for tid in planned_ids]
 n=len(rows);assessed=sum(x['answer_success'] is not None for x in rows);unknown=n-assessed
 result={'planned':n,'assessed':assessed,'unknown':unknown}
 for name in ['answer_success','supported_success']:
  k=sum(x[name] is True for x in rows)
  result[name+'_count']=k;result[name+'_rate_assessed']=k/assessed if assessed else None
  result[name+'_planned_bounds']=[k/n,(k+unknown)/n] if n else None
 from collections import Counter
 result['failure_categories']=dict(Counter(x['failure_category'] for x in rows))
 return result
