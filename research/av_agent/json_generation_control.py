"""Narrow optional generation control; no post-generation text modification.

Suppresses the observed compound `}.` token and stops after a complete JSON
object. This is not a full grammar/schema constraint or a semantic validator.
Only one sequence is supported by the fixed six-input smoke runner.
"""
import json
from transformers import StoppingCriteria,StoppingCriteriaList

def _unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result

def _reject_constant(value):
    raise ValueError('non-finite JSON constant')

class CompleteJSONObject(StoppingCriteria):
    def __init__(self,tokenizer,prompt_length):
        self.tokenizer=tokenizer
        self.prompt_length=prompt_length
        self.stopped_on_complete_object=False
    def __call__(self,input_ids,scores,**kwargs):
        if input_ids.ndim!=2 or input_ids.shape[0]!=1:
            raise ValueError('one generated sequence required')
        raw=self.tokenizer.decode(input_ids[0,self.prompt_length:].tolist(),
                                  skip_special_tokens=True,clean_up_tokenization_spaces=False)
        try:value=json.loads(raw,object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
        except (ValueError,TypeError,RecursionError):done=False
        else:done=isinstance(value,dict)
        self.stopped_on_complete_object=done
        return done

def generation_controls(tokenizer,prompt_length):
    if isinstance(prompt_length,bool) or not isinstance(prompt_length,int) or prompt_length<0:
        raise ValueError('nonnegative prompt token count required')
    token_ids=tokenizer.encode('}.',add_special_tokens=False)
    if len(token_ids)!=1 or tokenizer.decode(token_ids,clean_up_tokenization_spaces=False)!='}.':
        raise ValueError('verified compound trailing-period token required')
    stop=CompleteJSONObject(tokenizer,prompt_length)
    controls={'thinker_bad_words_ids':[token_ids],
              'thinker_stopping_criteria':StoppingCriteriaList([stop])}
    receipt={'policy':'json_object_v1','suppressed_token_ids':token_ids,
             'suppressed_token_text':'}.','stop_rule':'first full generated text that strictly decodes to a JSON object',
             'prompt_tokens_excluded':prompt_length,'raw_output_repair':False,
             'scope':'one observed compound token only; not full JSON grammar, schema, or semantics'}
    return controls,receipt
