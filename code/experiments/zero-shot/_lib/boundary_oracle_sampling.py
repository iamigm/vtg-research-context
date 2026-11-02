"""Fixed boundary sampler; no model and no performance-dependent choices."""
from boundary_oracle_inputs import *
import math,hashlib,random
BUDGETS=['full','half','quarter']
CONDITIONS=['uniform','gt_boundary','random_s42','random_s43']
CONFIG=dict(run_id=RUN,model='Qwen3-VL-8B-Instruct',revision=old.CONFIG['revision'],generation=dict(do_sample=False,max_new_tokens=512,repetition_penalty=1.0,seed=42),precision='BF16',backend='FlashAttention2',order='video_query',spatial_grid='original 2fps H/W at all budgets',budget_rule='F, 2*floor(F/4), 2*floor(F/8); unique source frames; no temporal padding',boundary_radius_seconds=1.,global_fraction=.5,random_seeds=[42,43],window_edges='clip to [0,T], do not shift',overlap='unique source IDs; ID-hash window order; borrow within union then nearest outside',bootstrap_replicates=20000,bootstrap_seed=42,scouts=False,pred=False,exploratory_hint='GT minus Uniform >=2pp and GT minus Random-two-seed mean >0; CI reported, not used as a hard screening gate',addition_gate='if no replacement budget has exploratory hint, run add-only diagnostic',addition_budget='retain every original Uniform frame; add the half-budget frame count at same H/W; explicitly greater visual-token budget',confirmation='if hint exists, freeze selected setting and report sample-size plan for new videos; do not auto-expand this quick screening run')

def even_ranks(ids,n):
    need(1<=n<=len(ids),'Rank selection budget invalid')
    if n==1:return [ids[(len(ids)-1)//2]]
    return [ids[(2*j*(len(ids)-1)+n-2)//(2*(n-1))] for j in range(n)]

def frame_budget(F,budget):return F if budget=='full' else max(2,2*(F//(4 if budget=='half' else 8)))

def plan(public,budget,condition,gt=None,add=False):
    need(set(public)=={'sample_key','duration','baseline_ids','ids','pts'},'GT-free public schema violated')
    key=public['sample_key'];T=public['duration'];base=public['baseline_ids'];ids=public['ids'];times=dict(zip(ids,public['pts']))
    n=frame_budget(len(base),budget);uniform=even_ranks(base,n)
    if condition=='uniform':return dict(sample_key=key,budget=budget,condition=condition,frame_ids=uniform,pts=[times[i] for i in uniform],anchors=uniform,focus=[],fill=[],windows=[],F=n,fill_fraction=0.,mode='replace')
    if condition=='gt_boundary':need(gt is not None,'Missing isolated GT boundary');centers=list(gt)
    else:
        seed=int(condition[-2:]);rng=random.Random(int.from_bytes(hashlib.sha256(f'{RUN}|{key}|{seed}'.encode()).digest()[:8],'big'));centers=sorted([rng.random()*T,rng.random()*T])
    windows=[[max(0,c-1),min(T,c+1)] for c in centers]
    need(all(a<=b for a,b in windows),'Invalid focus windows')
    anchors=base[:] if add else even_ranks(uniform,n//2)
    K=frame_budget(len(base),'half') if add else n-len(anchors)
    first=int.from_bytes(hashlib.sha256(f'{RUN}|{key}|allocation'.encode()).digest()[:8],'big')%2
    order=[first,1-first];quotas=[K//2,K//2];quotas[first]+=K%2;used=set(anchors);focus=[]
    for w in order:
        a,b=windows[w];candidates=[i for i in ids if i not in used and a<=times[i]<=b];q=min(quotas[w],len(candidates))
        chosen=[candidates[math.floor((j+.5)*len(candidates)/q)] for j in range(q)] if q else []
        focus.extend(chosen);used.update(chosen)
    available=[i for i in ids if i not in used and any(a<=times[i]<=b for a,b in windows)];extra=min(K-len(focus),len(available))
    borrow=[available[math.floor((j+.5)*len(available)/extra)] for j in range(extra)] if extra else []
    focus+=borrow;used.update(borrow)
    remaining=sorted((min(max(a-times[i],0,times[i]-b) for a,b in windows),i) for i in ids if i not in used)
    fill=[i for _,i in remaining[:K-len(focus)]];selected=sorted(anchors+focus+fill);need(len(selected)==len(anchors)+K and len(set(selected))==len(selected),'Distinct source-frame budget failure')
    if add:need(set(base)<=set(selected),'Add-only condition dropped an original frame')
    return dict(sample_key=key,budget=budget,condition=condition,frame_ids=selected,pts=[times[i] for i in selected],anchors=anchors,focus=focus,fill=fill,windows=windows,centers=centers,window_order=order,requested_quotas=quotas,borrowed=len(borrow),F=len(selected),fill_fraction=len(fill)/K,mode='add' if add else 'replace',overlap=windows[0][1]>=windows[1][0],clipped=any(c<1 or c>T-1 for c in centers))

def fixtures():
    p=dict(sample_key='fixture',duration=10.,baseline_ids=list(range(0,100,5)),ids=list(range(100)),pts=[i/10 for i in range(100)])
    out=[]
    for budget in BUDGETS:
        for gt in [[0,1],[9,10],[4,4.2],[4,6]]:
            r=plan(p,budget,'gt_boundary',gt);need(r['F']==frame_budget(20,budget),'Fixture budget differs');out.append(r)
        a=plan(p,budget,'random_s42');need(old.encoded(a)==old.encoded(plan(copy.deepcopy(p),budget,'random_s42')),'Random plan not reproducible')
    add=plan(p,'full','gt_boundary',[0,.1],True);need(set(p['baseline_ids'])<=set(add['frame_ids']),'Add fixture removed frame')
    return dict(passed=True,plans=out,add=add)
