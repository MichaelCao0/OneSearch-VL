"""Evidence-bound compositional QA checks; graph references never enter actor inputs."""
from __future__ import annotations
import math
import re
import unicodedata
from urllib.parse import urlsplit

OPERATIONS = {'lookup','multi_hop','join','arithmetic','comparison','conditional_count'}

def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).casefold().strip()

def number(value):
    if isinstance(value,bool): raise ValueError('Boolean is not a numeric operand')
    text=str(value).replace(',','').strip()
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?',text): raise ValueError('Explicit numeric tail required')
    x=float(text)
    if not math.isfinite(x): raise ValueError('Finite operand required')
    return x

def validate_anchors(value,duration):
    if not isinstance(value,dict) or not isinstance(value.get('anchors'),list) or not 1<=len(value['anchors'])<=8:
        raise ValueError('One to eight native-media anchors required')
    ids=set()
    for a in value['anchors']:
        if not isinstance(a,dict) or not all(isinstance(a.get(k),str) and a[k].strip() for k in ['id','observation','entity_hint']):
            raise ValueError('Anchor identity and actual observation required')
        if a['id'] in ids or a.get('modality') not in {'visual','audio','joint'}: raise ValueError('Distinct modality-bound anchors required')
        ids.add(a['id'])
        start,end=a.get('start_s'),a.get('end_s')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in [start,end]) or not 0<=start<end<=duration+.15:
            raise ValueError('Anchor outside actual media timeline')
        queries=a.get('search_queries')
        if not isinstance(queries,list) or not 1<=len(queries)<=2 or any(not isinstance(q,str) or not 1<=len(q.strip())<=128 for q in queries):
            raise ValueError('Bounded real search queries required')
    return value

def validate_graph(value,anchors,pages):
    if not isinstance(value,dict) or not isinstance(value.get('entities'),list) or not isinstance(value.get('facts'),list):
        raise ValueError('Entities and source-backed facts required')
    anchor_ids={a['id'] for a in anchors['anchors']}; entities={};facts={}
    for e in value['entities']:
        if not isinstance(e,dict) or not isinstance(e.get('id'),str) or not isinstance(e.get('name'),str) or not e['name'].strip() or e['id'] in entities:
            raise ValueError('Distinct named graph entities required')
        refs=e.get('anchor_ids',[])
        if not isinstance(refs,list) or set(refs)-anchor_ids:raise ValueError('Entity refers to absent media anchor')
        if refs and (e.get('binding_verified') is not True or not str(e.get('binding_reason','')).strip()):
            raise ValueError('Entity identity must be checked against actual AV and retrieved evidence')
        entities[e['id']]=e
    for f in value['facts']:
        if not isinstance(f,dict) or not all(isinstance(f.get(k),str) and f[k].strip() for k in ['id','head','relation','tail','source_id','quote']):
            raise ValueError('Complete atomic fact required')
        if f['id'] in facts or f['head'] not in entities or f.get('tail_kind') not in {'entity','number','text'}:
            raise ValueError('Invalid fact identity or endpoints')
        if f['tail_kind']=='entity' and f['tail'] not in entities:raise ValueError('Unknown tail entity')
        if f['tail_kind']=='number':number(f['tail'])
        p=pages.get(f['source_id'])
        if not p or p.get('kind')!='fetched_web_body' or len(norm(f['quote']))<6 or norm(f['quote']) not in norm(p['content']):
            raise ValueError('Quote must match actual fetched webpage body; search summary alone is insufficient')
        if not urlsplit(p['url']).hostname:raise ValueError('Fact lacks retrieved source URL')
        facts[f['id']]=f
    if not facts:raise ValueError('No verified external facts')
    return value

def validate_task(task,graph,anchors):
    entities={e['id']:e for e in graph['entities']}; facts={f['id']:f for f in graph['facts']}
    if not isinstance(task,dict) or task.get('operation') not in OPERATIONS:raise ValueError('Known research operation required')
    q=task.get('question');answers=task.get('accepted_answers')
    if not isinstance(q,str) or len(q.strip())<12 or not isinstance(answers,list) or not answers or any(not isinstance(a,str) or not a.strip() for a in answers):
        raise ValueError('Question and explicit accepted answers required')
    if any(len(norm(a))>=3 and norm(a) in norm(q) for a in answers):raise ValueError('Question leaks answer')
    paths=task.get('fact_paths')
    if not isinstance(paths,list) or not paths or any(not isinstance(p,list) or not p or len(set(p))!=len(p) or set(p)-facts.keys() for p in paths):
        raise ValueError('Explicit valid fact paths required')
    roots=set();used=set();intermediate=set()
    for path in paths:
        first=facts[path[0]];roots.update(entities[first['head']].get('anchor_ids',[]))
        if not entities[first['head']].get('anchor_ids'):raise ValueError('Each path must start at a media-bound entity')
        for a,b in zip(path,path[1:]):
            if facts[a]['tail_kind']!='entity' or facts[a]['tail']!=facts[b]['head']:raise ValueError('Broken multi-hop fact chain')
            intermediate.add(facts[a]['tail'])
        used.update(path)
    declared=task.get('anchor_ids')
    if not isinstance(declared,list) or set(declared)!=roots or roots-{a['id'] for a in anchors['anchors']}:
        raise ValueError('Task media anchors differ from evidence paths')
    for e in {facts[p[0]]['head'] for p in paths}|intermediate:
        name=entities[e]['name']
        if len(norm(name))>=4 and norm(name) in norm(q):raise ValueError('Explicit anchor or intermediate entity leaked in question')
    op=task['operation'];final=task.get('final_operation',{})
    inputs=final.get('input_fact_ids',[])
    if not isinstance(inputs,list) or set(inputs)!={p[-1] for p in paths}:raise ValueError('Final operation must consume exactly the fact-path outputs')
    kind=final.get('kind');expected=None
    if op=='lookup':
        if len(paths)!=1 or len(paths[0])!=1 or kind!='lookup':raise ValueError('Lookup structure mismatch')
    elif op=='multi_hop':
        if len(paths)!=1 or len(paths[0])<2 or kind!='lookup':raise ValueError('Multi-hop requires continuous external relation chain')
    else:
        if len(paths)<2 or len(inputs)<2:raise ValueError('Composition requires at least two evidence branches')
        if op in {'join','arithmetic','comparison','conditional_count'} and (len(roots)<2 or len({facts[p[0]]['head'] for p in paths})<2):
            raise ValueError('Multi-anchor operations require distinct media anchors')
    if op in {'lookup','multi_hop'}:
        f=facts[inputs[0]];expected=entities[f['tail']]['name'] if f['tail_kind']=='entity' else f['tail']
        if norm(expected) not in {norm(a) for a in answers}:raise ValueError('Answer differs from terminal fact')
    if op in {'arithmetic','comparison','conditional_count'}:
        fs=[facts[i] for i in inputs]
        if op in {'arithmetic','comparison'}:
            if len({norm(f['relation']) for f in fs})!=1 or len({norm(f.get('unit','')) for f in fs})!=1:
                raise ValueError('Comparable numeric attribute and compatible units required')
            values=[number(f['tail']) for f in fs]
        if op=='arithmetic':
            if kind=='absolute_difference' and len(values)==2:expected=abs(values[0]-values[1])
            elif kind=='sum':expected=sum(values)
            else:raise ValueError('Unsupported deterministic arithmetic')
            if not any(abs(number(a)-expected)<1e-7 for a in answers if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',a.replace(',','').strip())):
                raise ValueError('Derived arithmetic answer is wrong')
        elif op=='comparison':
            if kind not in {'argmax','argmin'} or len(set(values))!=len(values):raise ValueError('Unambiguous comparison required')
            best=(max if kind=='argmax' else min)(range(len(values)),key=values.__getitem__)
            expected=entities[fs[best]['head']]['name']
            if norm(expected) not in {norm(a) for a in answers}:raise ValueError('Comparison winner differs from answer')
        else:
            condition=final.get('condition_value')
            if kind=='count_equals':expected=sum(norm(f['tail'])==norm(condition) for f in fs)
            elif kind=='count_at_least':expected=sum(number(f['tail'])>=number(condition) for f in fs)
            else:raise ValueError('Explicit counting predicate required')
            if str(expected) not in answers:raise ValueError('Conditional count differs from answer')
    elif op=='join' and kind!='join':raise ValueError('Join operation mismatch')
    return {'operation':op,'max_fact_hops':max(map(len,paths)),'required_facts':len(used),'anchors':len(roots),'computed_answer':expected}

def actor_case(task,source):
    # Closed construction prevents private graph/metadata/reference contamination.
    return {'case_id':task['task_id'],'question':task['question'],'source_path':source['source_path'],'source_sha256':source['source_sha256']}
