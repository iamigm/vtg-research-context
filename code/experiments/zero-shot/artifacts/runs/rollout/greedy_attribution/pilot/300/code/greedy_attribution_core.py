"""Cumulative greedy attribution: frozen configuration, inputs and scalar metrics."""
import collections, datetime, hashlib, json, os, re
from pathlib import Path

ROOT = Path('/home/irteam/experiments/zero-shot')
FAMILY = 'rollout/greedy_attribution'
BASE = ROOT / 'artifacts/runs' / FAMILY / 'pilot'
RESULTS = ROOT / 'results' / FAMILY
SOURCE = ROOT / 'artifacts/runs/rollout/deterministic/pilot/900'
COHORT = ROOT / 'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
MODEL = Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
GROUPS = ['visual', 'timestamp', 'query', 'instruction', 'special', 'generated']
CONFIG = dict(family=FAMILY, stages=[300,600,900], model=str(MODEL), input_order='video_query',
    frame_sampling_fps=2, input_policy='exact deterministic source tensors; hash checked',
    candidate_top_k=30, candidate_scope='frozen original top-30 tokens at each saved greedy step; outside unmeasured',
    screen_epsilon=.03125, screen_directions=['video_pixels','timestamp','query','instruction','special','generated'],
    screening='central finite differences of original top-30 logits under one source-group scaling at a time; raw softmax probabilities are not renormalized',
    perturbation_domain='normalized video pixels and text input embeddings; internal diagnostic, not natural query edits',
    current_min_probability=.01, current_max_logit_gap=4.605170185988092,
    potential_max_screen_gap=1.0, screen_candidates_per_direction=2,
    detailed_branch_targets=4, chosen_targets='every greedy output token including format and EOS',
    endpoint_targets=True, temperature=1, checkpoint_input_tokens=6000,
    replay_probability_tolerance=.001, probability_precision='FP32 softmax of native BF16 logits',
    replay='original multimodal prefill then single-token decoding; immutable differentiable KV chunks; exact argmax required',
    mode_endpoint_tolerance_seconds=1.0, mode_relative_tolerance=.02,
    mode_policy='reference saved deterministic continuations sharing the exact branch; unobserved mode remains unknown',
    interpretation='local sensitivity and signed signal scaling derivatives; not causal evidence or full mode probability',
    seed=42)

def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def storage():
    for n in (900,600,300):
        path=BASE/str(n)
        if (path/'metadata/config.json').exists(): return path
    return BASE/'300'
def iou(a,b):
    if not a or not b:return None
    return max(0,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0]))

def context_map(ids, tokenizer, prompt_length, row, grid, timestamps):
    pieces=tokenizer.batch_decode([[int(i)] for i in ids],clean_up_tokenization_spaces=False)
    text=''.join(pieces[:prompt_length]); qstart=text.rfind(row['query'])
    if qstart<0: raise ValueError('Query text not found in exact input token decoding')
    qr=(qstart,qstart+len(row['query'])); times=[m.span() for m in re.finditer(r'<[0-9:.]+ seconds>',text)]
    groups=[];offset=0; specials=set(tokenizer.all_special_ids)
    for i,(ident,piece) in enumerate(zip(ids,pieces)):
        end=offset+len(piece)
        if i>=prompt_length: group='generated'
        elif ident==151656: group='visual'
        elif ident in specials: group='special'
        elif offset<qr[1] and end>qr[0]:group='query'
        elif any(offset<b and end>a for a,b in times):group='timestamp'
        else:group='instruction'
        groups.append(group);offset=end
    t,h,w=grid; visual=[i for i,g in enumerate(groups) if g=='visual']; assert len(visual)==t*h*w//4
    mapping=[]
    for j,pos in enumerate(visual):
        frame=j//((h//2)*(w//2)); spatial=j%((h//2)*(w//2))
        ts=timestamps[2*frame:2*frame+2]
        mapping.append(dict(position=pos,frame_pair=frame,row=spatial//(w//2),col=spatial%(w//2),timestamp=sum(ts)/len(ts)))
    return dict(ids=list(ids),tokens=pieces,groups=groups,visual_positions=visual,visual_map=mapping)

def source_mode(record, step, token, prefix, duration):
    out=[]; greedy=record['result']['candidates'][0]['parsed']
    base=[greedy['start'],greedy['end']] if greedy and greedy['positive'] else None
    tol=max(CONFIG['mode_endpoint_tolerance_seconds'],CONFIG['mode_relative_tolerance']*duration)
    for c in record['result']['candidates']:
        if len(c['ids'])<=step or c['ids'][:step]!=prefix or c['ids'][step]!=token:continue
        p=c.get('parsed');span=[p['start'],p['end']] if p and p.get('positive') else None
        out.append(dict(candidate=c['index'],span=span,answer=c['answer'],log_probability=c['log_probability'],
            separated=bool(base and span and max(abs(span[0]-base[0]),abs(span[1]-base[1]))>tol)))
    return dict(status='observed_continuations' if out else 'unobserved',endpoint_tolerance=tol,
                candidates=out,distinct_mode_candidate=any(c['separated'] for c in out))
