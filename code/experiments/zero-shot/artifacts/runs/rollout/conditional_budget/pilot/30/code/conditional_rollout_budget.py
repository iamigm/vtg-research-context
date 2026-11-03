"""Deterministic joint start/end probability-tree budget calibration.

No sampling. Probabilities are raw full-vocabulary probabilities, conditional on
the greedily generated answer preamble, not renormalized by the numeric grammar.
"""
import os
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse, collections, datetime, fcntl, hashlib, heapq, json, math
import re, subprocess, sys, threading, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'conditional_rollout_budget_pilot30_v1'
OUT = ROOT/'artifacts/runs'/RUN
REPORT = ROOT/'results'/RUN
MODEL = Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
CHECKPOINTS = [8, 16, 32, 64, 128, 256, 512]
CONFIG = dict(run=RUN, model=str(MODEL), diagnostic_queries=30,
    selection='first 10 per dataset in frozen pilot300 order; no outcome selection',
    fps=2, target_pixels=100352, input_order='video_query',
    input_source='fps_spatial_control_pilot300_v1 fixed H,W/PTS cache',
    decoding='deterministic best-first token tree, no stochastic calls',
    distribution='raw full vocabulary softmax, temperature=1, no penalties',
    anchor='greedy generated preamble immediately before first digit',
    grammar='nonnegative integer or one-decimal S - E seconds; max digits derived from duration, at least 4; max 32 tokens',
    mass='conditional on exact preamble; invalid grammar mass is NOT renormalized',
    priorities=['joint', 'balanced'], checkpoints=CHECKPOINTS,
    max_expansions=512, max_suffix_tokens=32,
    resource_gate='existing FPS run complete AND >=60 GiB free GPU; other jobs never stopped')

def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Progress:
    def __init__(self):
        self.started=time.monotonic();self.lock=threading.RLock();self.stop=threading.Event()
        self.s=dict(state='running',stage='initializing',completed=0,total=30,
                    model='qwen3-instruct',counts={},eta_seconds=None,
                    eta_scope='inference only; unknown until first completed query; excludes GPU queue')
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def update(self,**kw):
        with self.lock:
            self.s.update(kw)
            self.s.update(timestamp=stamp(),elapsed_seconds=time.monotonic()-self.started,
                          percentage=100*self.s['completed']/self.s['total'])
            write(OUT/'metadata/status.json',self.s)
            print(json.dumps(self.s,ensure_ascii=False),flush=True)
    def heartbeat(self):
        while not self.stop.wait(20):self.update()

def setup():
    cohort=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_300/samples.json'
    allrows=read(cohort);counts=collections.Counter();rows=[]
    for r in allrows:
        if counts[r['dataset']]<10:rows.append(r);counts[r['dataset']]+=1
    assert len(rows)==30 and sorted(counts.values())==[10,10,10]
    for category in ['metadata','logs','code','trees','metrics','validation','inputs']:
        (OUT/category).mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    for name,value in [('config.json',CONFIG),('cohort.json',rows)]:
        p=OUT/'metadata'/name
        if p.exists():assert read(p)==value, f'Frozen configuration changed: {p}'
        else:write(p,value)
    source=Path(__file__)
    (OUT/'code'/source.name).write_bytes(source.read_bytes())
    dep=source.with_name('fps_spatial_control.py')
    (OUT/'code'/dep.name).write_bytes(dep.read_bytes())
    write(OUT/'metadata/provenance.json',dict(cohort=str(cohort),cohort_sha256=sha(cohort),
          code_sha256=sha(source),input_code_sha256=sha(dep),created_at=stamp(),
          cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
          memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
          backbone_manifest_sha256=sha('/home/irteam/backbones/manifests/registry.json')))
    return rows

def grammar(duration):
    import regex
    digits=max(4,len(str(math.ceil(duration))))
    number=rf'(?:0|[1-9][0-9]{{0,{digits-1}}})(?:\.[0-9])?'
    return regex.compile(rf'(?P<start>{number})\s{{0,2}}-\s{{0,2}}(?P<end>{number}) seconds'), re.compile(rf'^({number})\s{{0,2}}-\s{{0,2}}$')

class Evaluator:
    """One mutable KV cache; crop/replay shared text prefixes without redoing video."""
    def __init__(self,model,inputs,torch):
        self.model=model;self.inputs=inputs;self.torch=torch;self.n=inputs.input_ids.shape[1]
        self.path=();self.logits={};self.calls=0;self.text_tokens=0
        t=time.monotonic()
        with torch.inference_mode():o=model(**inputs,use_cache=True,logits_to_keep=1)
        self.cache=o.past_key_values;self.logits[()]=o.logits[0,-1].float().cpu()
        self.prefill_seconds=time.monotonic()-t
        del o
    def get(self,path):
        path=tuple(path)
        if path in self.logits:return self.logits[path]
        common=0
        for a,b in zip(self.path,path):
            if a!=b:break
            common+=1
        # An ancestor's logits may have been evicted for fair policy costs.
        # Recompute its final token rather than querying an empty suffix.
        if common==len(path):common=max(0,len(path)-1)
        self.cache.crop(self.n+common)
        tokens=path[common:];assert tokens
        torch=self.torch
        # Always replay one text token per forward. Variable suffix batch shapes
        # otherwise introduce BF16 differences that depend on traversal order.
        for j,token in enumerate(tokens,common):
            with torch.inference_mode():
                o=self.model(input_ids=torch.tensor([[token]],device='cuda'),
                    attention_mask=torch.ones((1,self.n+j+1),device='cuda',dtype=self.inputs.attention_mask.dtype),
                    past_key_values=self.cache,cache_position=torch.tensor([self.n+j],device='cuda'),
                    use_cache=True,logits_to_keep=1)
            self.cache=o.past_key_values;self.path=path[:j+1]
            self.logits[self.path]=o.logits[0,-1].float().cpu();self.calls+=1;self.text_tokens+=1
            del o
        return self.logits[path]

def tree(evaluator,tokenizer,anchor,duration,policy,progress):
    """Each start key is an exact token prefix through its end delimiter/space.

    Starts with different whitespace/tokenization remain separate conditioning
    contexts. Parsed numeric values are only merged in reporting after inference.
    """
    torch=evaluator.torch;pattern,start_pattern=grammar(duration)
    vocab=[]
    for i in range(len(tokenizer)):
        if i in tokenizer.all_special_ids:continue
        s=tokenizer.decode([i],clean_up_tokenization_spaces=False)
        if s and re.fullmatch(r'[0-9.\- seconds]+',s):vocab.append((i,s))
    frontier=[(-1.,(),'',None)]
    starts={};leaves=[];discard=0.;truncated=0.;snapshots=[];seen=set()
    initial_calls=evaluator.calls;initial_tokens=evaluator.text_tokens;began=time.monotonic()
    expanded_start=expanded_end=0
    for expanded in range(1,CONFIG['max_expansions']+1):
        if not frontier:break
        if policy=='balanced':
            # Soften repeated deepening under the same start, weighted by mass.
            # This does not guarantee expansion of every low-probability start.
            def priority(entry):
                mass=-entry[0];key=entry[3]
                return mass/(1+starts[key]['expansions']) if key is not None else mass
            index=max(range(len(frontier)),key=lambda i:(priority(frontier[i]),frontier[i][0],frontier[i][1]))
            entry=frontier.pop(index);heapq.heapify(frontier)
        else:entry=heapq.heappop(frontier)
        negmass,path,text,key=entry;mass=-negmass
        assert path not in seen;seen.add(path)
        if key is None:expanded_start+=1
        else:expanded_end+=1;starts[key]['expansions']+=1
        probs=torch.softmax(evaluator.get(tuple(anchor)+path),-1)
        accepted=0.
        for token,piece in vocab:
            newtext=text+piece
            match=pattern.fullmatch(newtext,partial=True)
            if match is None:continue
            probability=float(probs[token]);accepted+=probability
            if probability<=0:continue
            p=mass*probability;child=path+(token,);childkey=key
            # Detect the start-complete boundary. Tokens spanning this boundary
            # are recorded as their own conditional context, never split falsely.
            if childkey is None and '-' in newtext:
                boundary=newtext.index('-')+1
                while boundary<len(newtext) and newtext[boundary]==' ':boundary+=1
                numeric_end=newtext[boundary:]
                childkey=','.join(map(str,child))
                starts[childkey]=dict(prefix_ids=list(child),prefix_text=newtext,
                    start=float(newtext[:newtext.index('-')].strip()),mass=p,
                    boundary_spanning=bool(numeric_end),expansions=0)
            if not match.partial:
                assert childkey is not None
                leaves.append(dict(ids=list(child),text=newtext,start=float(match['start']),
                    end=float(match['end']),mass=p,start_key=childkey,found_at=expanded,
                    valid=0<=float(match['start'])<float(match['end'])<=duration))
            elif len(child)>=CONFIG['max_suffix_tokens']:truncated+=p
            elif p>0:heapq.heappush(frontier,(-p,child,newtext,childkey))
        discard+=mass*max(0.,1-accepted)
        if expanded in CHECKPOINTS or not frontier:
            snap=dict(expansions=expanded,start_expansions=expanded_start,end_expansions=expanded_end,
                start_mass=sum(x['mass'] for x in starts.values()),
                joint_mass=sum(x['mass'] for x in leaves),
                frontier_mass=sum(-x[0] for x in frontier),grammar_rejected_mass=discard,
                token_limit_mass=truncated,starts=len(starts),pairs=len(leaves),
                valid_pairs=sum(x['valid'] for x in leaves),
                incremental_forward_calls=evaluator.calls-initial_calls,
                incremental_decoder_tokens=evaluator.text_tokens-initial_tokens,
                elapsed_seconds=time.monotonic()-began)
            total=snap['joint_mass']+snap['frontier_mass']+discard+truncated
            assert abs(total-1)<2e-5,('Mass conservation',total)
            assert snap['joint_mass']<=snap['start_mass']+2e-5
            snapshots.append(snap)
            progress.update(stage='tree_search',policy=policy,expansions=expanded,
                joint_mass=snap['joint_mass'],start_mass=snap['start_mass'])
    return dict(policy=policy,snapshots=snapshots,starts=starts,leaves=leaves,
        conditioning='exact prefix tokens including start and the first delimiter-bearing token',
        probability_scope='conditional on fixed greedy preamble; raw softmax; no grammar renormalization',
        cost_scope='prefill shared; priority-specific incremental costs depend on evaluator cache reuse')

def validate_cache(evaluator,inputs,anchor,tokenizer,torch):
    # Validate branching rollback, not only the original greedy trajectory.
    prefix=tuple(anchor)
    digit=tokenizer.encode('9',add_special_tokens=False)
    paths=[prefix+tuple(digit),prefix+tuple(tokenizer.encode('1',add_special_tokens=False))]
    audits=[]
    for path in paths:
        actual=evaluator.get(path)
        kw=dict(inputs);kw['input_ids']=torch.cat([inputs.input_ids,torch.tensor([path],device='cuda')],dim=1)
        kw['attention_mask']=torch.ones_like(kw['input_ids'])
        with torch.inference_mode():fresh=evaluator.model(**kw,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
        # Fresh prefill recalculates rope_deltas for the same video/text context.
        ap=torch.softmax(actual,-1);bp=torch.softmax(fresh,-1)
        audit=dict(prefix=list(path),top1_equal=int(actual.argmax())==int(fresh.argmax()),
            full_prefill_probability_l1=float((ap-bp).abs().sum()))
        reference=Evaluator(evaluator.model,inputs,torch)
        expected=reference.get(path)
        rp=torch.softmax(expected,-1)
        audit.update(rollback_top1_equal=int(actual.argmax())==int(expected.argmax()),
                     rollback_probability_l1=float((ap-rp).abs().sum()),
                     validation='rollback vs clean original multimodal prefill + identical single-token replay')
        assert audit['rollback_top1_equal'] and audit['rollback_probability_l1']<.001,audit
        del reference
        audits.append(audit)
    return audits

def wait_resources(progress):
    prior=ROOT/'artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json'
    while True:
        state=read(prior) if prior.exists() else {}
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        free=int(result.stdout.splitlines()[0])
        release=OUT/'metadata/resource_release.json'
        explicitly_released=release.exists() and read(release).get('authorized') is True
        prior_complete=state.get('state')=='complete' and state.get('stage')=='complete'
        if (prior_complete or explicitly_released) and free>=60*1024:return
        progress.update(state='waiting',stage='waiting_for_gpu',free_gpu_mib=free,
                        preceding_run=state.get('state'),preceding_completed=state.get('completed'))
        time.sleep(20)

def run(rows,progress):
    wait_resources(progress)
    import fps_spatial_control as source
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.set_num_threads(2)
    progress.update(state='running',stage='loading_model')
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,do_resize=False)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
    inference_start=time.monotonic();counts=collections.Counter();completed=0
    for row in rows:
        destination=OUT/'trees'/(row['sample_key']+'.json')
        if destination.exists():completed+=1;counts[row['dataset']]+=1;continue
        progress.update(stage='preparing_input',sample_key=row['sample_key'])
        key=hashlib.sha256(row['video_path'].encode()).hexdigest()
        cachepath=source.OUT/'media'/key/'metadata.json';cache=read(cachepath)
        public={k:row[k] for k in ['sample_key','dataset','video_path','query','duration']}
        cpu,detail=source.prepare_input(public,cache,2,'video_query',proc)
        write(OUT/'inputs'/(row['sample_key']+'.json'),detail)
        inputs=cpu.to('cuda');ev=Evaluator(model,inputs,torch)
        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        greedy=[];anchor=None
        for i in range(128):
            token=int(ev.get(greedy).argmax());piece=proc.tokenizer.decode([token],clean_up_tokenization_spaces=False)
            if anchor is None and re.search(r'[0-9]',piece):
                assert re.match(r'[0-9]',piece),'First numeric token spans preamble boundary'
                anchor=greedy[:]
            greedy.append(token)
            if token in eos:break
        assert anchor is not None,'No numeric answer; diagnostic failed explicitly'
        audits=validate_cache(ev,inputs,anchor,proc.tokenizer,torch) if completed==0 else None
        if audits:write(OUT/'validation/cache_branching.json',dict(passed=True,audits=audits,sample_key=row['sample_key']))
        # Reset the evaluator for each policy so measured compute is comparable.
        trees=[];costs=[]
        del ev;torch.cuda.empty_cache()
        for policy in CONFIG['priorities']:
            ev=Evaluator(model,inputs,torch)
            t=tree(ev,proc.tokenizer,anchor,row['duration'],policy,progress)
            costs.append(dict(policy=policy,prefill_seconds=ev.prefill_seconds,
                              decoder_tokens=ev.text_tokens,forward_calls=ev.calls))
            trees.append(t)
            del ev;torch.cuda.empty_cache()
        write(destination,dict(sample=row,anchor_ids=anchor,
            anchor_text=proc.tokenizer.decode(anchor),greedy_ids=greedy,
            greedy_answer=proc.tokenizer.decode(greedy,skip_special_tokens=True),
            input_hashes=detail['hashes'],trees=trees,costs=costs))
        completed+=1;counts[row['dataset']]+=1
        elapsed=time.monotonic()-inference_start
        progress.update(stage='generation',completed=completed,counts=dict(counts),
            eta_seconds=elapsed/completed*(len(rows)-completed),
            eta_scope='remaining diagnostic inference at observed query rate; excludes queue and report')
        del inputs,cpu;torch.cuda.empty_cache()
    progress.update(stage='reporting')
    report()
    progress.update(state='complete',stage='complete',completed=len(rows),eta_seconds=0)

def iou(s,e,gt):
    return max(0.,min(e,gt[1])-max(s,gt[0]))/max(1e-12,max(e,gt[1])-min(s,gt[0]))

def report():
    import csv
    records=[read(p) for p in sorted((OUT/'trees').glob('*.json'))]
    lines=['# 시작 조건부 종료 탐색 예산 진단','',
        '확률은 greedy preamble에 조건부인 원래 full-vocabulary softmax이다. 숫자 문법 밖 질량을 재정규화하지 않는다. 시작·종료를 독립으로 가정하지 않는다.', '',
        '진단 표본은 고정 pilot300의 데이터셋별 첫 10개(총30개)이며 전체 벤치마크 평균으로 해석하지 않는다. 입력은 기존 FPS 실험의 고정 공간 2 FPS 입력을 재사용한다.', '',
        'S는 발견한 서로 겹치지 않는 시작-prefix 질량 합, J는 완성된 구간 질량 합이다. 1−J=(1−S)+(S−J): 첫 항은 아직 시작-prefix로 확보하지 못한 질량(형식 밖 포함), 두 번째는 확보한 시작 아래 아직 완성하지 못한 종료 질량이다. 대표 greedy 결과에 subtree 전체 질량을 부여하지 않는다.', '',
        '확장 수는 다음 토큰 분포를 조사한 고유 prefix 수다. decoder 토큰 수는 캐시 rollback 후 재계산한 토큰을 포함한다. prefill 시간은 별도다. 두 정책은 각각 독립 evaluator로 계산한다.', '',
        'mIoU와 R@1은 GT로 최선 후보를 선택한 oracle 상한이다. 실제 선택 성능이 아니다. IoU=교집합/합집합, R@1@τ=최선 IoU≥τ인 질문 비율이며 높을수록 좋다. 유효 구간은 0≤s<e≤영상 길이다. 후보가 없으면 IoU=0이다.', '',
        '| 정책 | 확장 상한 | N | 시작 질량 S | 구간 질량 J | 시작 미확보 1−S | 종료 미완성 S−J | decoder 토큰 | oracle mIoU | R@.3 | R@.5 | R@.7 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for r in records:
        for t in r['trees']:
            for s in t['snapshots']:
                candidates=[x for x in t['leaves'] if x['found_at']<=s['expansions'] and x['valid']]
                best=max([iou(x['start'],x['end'],r['sample']['span'][0]) for x in candidates] or [0.])
                csvrows.append(dict(sample_key=r['sample']['sample_key'],dataset=r['sample']['dataset'],
                    duration=r['sample']['duration'],gt_ratio=(r['sample']['span'][0][1]-r['sample']['span'][0][0])/r['sample']['duration'],
                    policy=t['policy'],**s,oracle_iou=best))
    for policy in CONFIG['priorities']:
        for cap in CHECKPOINTS:
            group=[x for x in csvrows if x['policy']==policy and x['expansions']==cap]
            if not group:continue
            avg=lambda k:sum(x[k] for x in group)/len(group)
            S,J=avg('start_mass'),avg('joint_mass')
            recalls=[sum(x['oracle_iou']>=q for x in group)/len(group) for q in [.3,.5,.7]]
            lines.append(f'| {policy} | {cap} | {len(group)} | {S:.4f} | {J:.4f} | {1-S:.4f} | {S-J:.4f} | {avg("incremental_decoder_tokens"):.1f} | {avg("oracle_iou"):.4f} | '+ ' | '.join(f'{x:.4f}' for x in recalls)+' |')
    # Conditional end concentration for each exact start prefix. Only certify
    # K90 when observed raw conditional leaf mass reaches .90; never normalize tails.
    conditional=[];dependence=[]
    for r in records:
        for t in r['trees']:
            distributions={}
            for key,start in t['starts'].items():
                ends=collections.defaultdict(float)
                for leaf in t['leaves']:
                    if leaf['start_key']==key:ends[leaf['end']]+=leaf['mass']/start['mass']
                probs=sorted(ends.values(),reverse=True);acc=0.;k90=None
                for k,p in enumerate(probs,1):
                    acc+=p
                    if acc>=.9 and k90 is None:k90=k
                conditional.append(dict(sample_key=r['sample']['sample_key'],policy=t['policy'],
                    start_prefix=key,start=start['start'],start_mass=start['mass'],
                    conditional_end_covered_mass=sum(probs),end_candidates=len(ends),
                    k90_upper_bound=k90,unresolved_k90=k90 is None,
                    boundary_spanning=start['boundary_spanning']))
                distributions[key]=ends
            # Compare the two highest-mass distinct numeric starts. End measures
            # are subprobabilities; bound TV without normalizing away unseen tails.
            ranked=sorted(t['starts'],key=lambda k:(-t['starts'][k]['mass'],k))
            if ranked:
                a=ranked[0]
                b=next((k for k in ranked[1:] if t['starts'][k]['start']!=t['starts'][a]['start']),None)
                if b is not None:
                    da,db=distributions[a],distributions[b]
                    l1=sum(abs(da.get(e,0)-db.get(e,0)) for e in set(da)|set(db))
                    residual=max(0,1-sum(da.values()))+max(0,1-sum(db.values()))
                    dependence.append(dict(sample_key=r['sample']['sample_key'],policy=t['policy'],
                        start_a=t['starts'][a]['start'],start_b=t['starts'][b]['start'],
                        observed_end_mass_a=sum(da.values()),observed_end_mass_b=sum(db.values()),
                        conditional_end_tv_lower=max(0,(l1-residual)/2),
                        conditional_end_tv_upper=min(1,(l1+residual)/2)))
    write(OUT/'metrics/conditional_end.json',conditional)
    write(OUT/'metrics/start_end_dependence.json',dependence)
    lines+=['', '| 정책 | 완료 질문 | J≥.90 도달 | J≥.95 도달 | .90 도달 확장수 중앙값 | .95 도달 확장수 중앙값 |',
            '|---|---:|---:|---:|---:|---:|']
    import statistics
    for policy in CONFIG['priorities']:
        reached={q:[] for q in [.9,.95]}
        for r in records:
            t=next(x for x in r['trees'] if x['policy']==policy)
            for q in reached:
                hit=next((s['expansions'] for s in t['snapshots'] if s['joint_mass']>=q),None)
                if hit is not None:reached[q].append(hit)
        med=lambda q:str(statistics.median(reached[q])) if reached[q] else '미도달'
        lines.append(f'| {policy} | {len(records)} | {len(reached[.9])} | {len(reached[.95])} | {med(.9)} | {med(.95)} |')
    lines+=['', '도달 확장수는 기록 checkpoint 기준 상한이다. 미도달 질문을 중앙값 계산에서 제외했으므로 도달 질문 수와 함께 읽어야 한다.', '',
        '| 정책 | 서로 다른 시작 쌍 수 | TV 하한 평균 | TV 상한 평균 | 종료 K90 해결 / 시작 prefix 수 |',
        '|---|---:|---:|---:|---:|']
    for policy in CONFIG['priorities']:
        ds=[x for x in dependence if x['policy']==policy]
        cs=[x for x in conditional if x['policy']==policy]
        lo=sum(x['conditional_end_tv_lower'] for x in ds)/len(ds) if ds else 0
        hi=sum(x['conditional_end_tv_upper'] for x in ds)/len(ds) if ds else 0
        lines.append(f'| {policy} | {len(ds)} | {lo:.4f} | {hi:.4f} | {sum(not x["unresolved_k90"] for x in cs)} / {len(cs)} |')
    if csvrows:
        with (OUT/'metrics/budget.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines+=['','K90은 관측한 종료 후보가 조건부 확률 90%를 덮는 데 필요한 후보 수의 상한이다. 미탐색 후보가 있어 최솟값으로 확정하지 않으며, 90% 미달은 unresolved로 기록한다.', '',
        '서로 다른 시작 중 확률이 가장 큰 두 시작의 조건부 종료 분포를 TV 거리로 비교한다. TV=0.5Σ|p−q|이며 클수록 종료 조건부 분포 차이가 크다. 미탐색 꼬리를 재정규화하지 않고 TV의 하한·상한을 계산하므로 꼬리가 크면 결론이 미정일 수 있다. 조건부 분포 차이는 영상 grounding 또는 사건 혼동의 증거와는 구분한다.', '',
        f'[시작별 종료 분포 차이](../../artifacts/runs/{RUN}/metrics/start_end_dependence.json)', '',
        f'완료 원자료 {len(records)}/30. 시작별 종료 질량은 [조건부 종료 자료](../../artifacts/runs/{RUN}/metrics/conditional_end.json), 예산별 자료는 [CSV](../../artifacts/runs/{RUN}/metrics/budget.csv)에 있다.', '',
        '예산 권고는 raw joint 질량 0.90/0.95 달성 여부와 oracle 성능 포화, 계산량을 함께 보고 정한다. 최대 상한에서 미달하면 예산을 확보했다고 주장하지 않는다. 숫자 문법 제한 때문에 0.95가 도달 불가능할 수도 있다.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for r in records:
        for t in r['trees']:
            for s in t['snapshots']:
                assert abs(s['joint_mass']+s['frontier_mass']+s['grammar_rejected_mass']+s['token_limit_mass']-1)<2e-5
    write(OUT/'validation/report.json',dict(passed=True,records=len(records),mass_conservation=True,
        conditional_end_not_renormalized=True,all_30_complete=len(records)==30))
    subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)

def selftest():
    # Asymmetric conditional ends: independent multiplication would be wrong.
    starts=[.5,.4];ends=[[.9,.1],[.2,.8]]
    joint=[a*b for a,bs in zip(starts,ends) for b in bs]
    assert all(abs(a-b)<1e-12 for a,b in zip(joint,[.45,.05,.08,.32]))
    S=sum(starts);J=joint[0]+joint[3]
    assert abs((1-J)-((1-S)+(S-J)))<1e-12
    pattern,_=grammar(140)
    for text in ['1','119.0 - ','119.0 - 136.0 seconds']:
        assert pattern.fullmatch(text,partial=True) is not None,text
    for text in ['01','1.23','-1','1 - x']:
        assert pattern.fullmatch(text,partial=True) is None,text
    return dict(passed=True,joint_probabilities=joint,start_mass=S,joint_mass=J,
                start_uncovered=1-S,end_uncovered=S-J,grammar=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--report',action='store_true')
    args=ap.parse_args()
    if args.selftest:print(json.dumps(selftest()));return
    if args.report:report();return
    rows=setup()
    with (OUT/'metadata/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=Progress()
        try:
            write(OUT/'validation/cpu_selftest.json',selftest());run(rows,p)
        except BaseException:
            error=traceback.format_exc();p.update(state='failed',stage='failed',error=error);raise
        finally:p.stop.set()

if __name__=='__main__':main()
