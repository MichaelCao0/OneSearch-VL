"""Compare action+answer and answer-only supervision on identical verified tokens."""
def apply_supervision(labels,pairs,mode):
    if mode not in {'trajectory','answer_only'}:raise ValueError('Explicit supervision mode required')
    if not pairs:raise ValueError('At least one final assistant span required')
    result=list(labels);end=0
    for p in pairs:
        start=p['assistant_token_start'];count=p['assistant_token_count']
        if isinstance(start,bool) or isinstance(count,bool) or not isinstance(start,int) or not isinstance(count,int) or start<end or count<1 or start+count>len(labels):raise ValueError('Invalid assistant span')
        if any(x==-100 for x in labels[start:start+count]):raise ValueError('Assistant span contains ignored input tokens')
        end=start+count
    if mode=='answer_only':
        for p in pairs[:-1]:result[p['assistant_token_start']:p['assistant_token_start']+p['assistant_token_count']]=[-100]*p['assistant_token_count']
    return result
