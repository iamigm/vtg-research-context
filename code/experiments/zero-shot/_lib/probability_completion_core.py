"""GT-free proposal reuse, temporal separation, and scalar semantic diagnostics."""
from probability_budget_core import *
PARENT=OUT
OUT=PARENT/'end_to_end'
ANALYSIS='end_to_end'
BACKBONES=Path('/home/irteam/backbones/models')
EMBED_SOURCE=ROOT.parent/'similarity/_lib/vendor/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py'
SETTINGS=dict(analysis=ANALYSIS,queries=900,stages=[300,900],proposal_config_hash=digest(CONFIG),methods=METHODS,budgets=BUDGETS,
 verifier='Qwen2.5-VL-7B-Instruct',generator='Qwen3-VL-8B-Instruct',embedding='Qwen3-VL-Embedding-8B',
 review='all unique proposed clipped intervals and greedy; independent crop, no proposal probabilities, no other candidates, no GT',
 score='next-token native probabilities for 0=absent/wrong, 1=partial or loose, 2=complete and localized; conditional expectation (p1+2p2)/2; raw option mass also retained',
 crop='original 2 FPS selected frames inside interval; when none, original frame displayed at midpoint; preserve order, pad one frame to two; normalize crop timestamps',
 spatial='Qwen3 original selected-frame HxW; verifier nearest positive multiples of28, no frame-count reduction',
 generation='greedy, max64 tokens; same original full-video/query KV, original greedy response, all candidates, no semantic scores; proposal-only control and review-conditioned answer',
 semantic='query-free crops; Instruct final-layer visual-token mean and last nonpadding token; Embedding official last-nonpadding pooling; L2 normalization; no captions, no pruning, no persistent vectors',
 temporal='Gaussian KDE at exact endpoint support, h=1s; 8-nearest support graph within4h; edge saddle=min(endpoint densities, midpoint density); merge if saddle/min(peak densities)>=.5; sensitivity h=.5/2',
 entropy='exact clipped interval atoms, not KDE or mode bins; observed valid mass only',batch_tokens=16384,batch_size=8,bootstrap_seed=42)
REFERENCES=[dict(name='VideoMind',url='https://github.com/yeliudev/VideoMind',use='localize, verify isolated moments, synthesize; no trained LoRA or paper performance reproduced'),dict(name='VideoAgent',url='https://github.com/wxh1996/VideoAgent',use='visual evidence inspection before answering; no API, caption pipeline, or reported results reproduced'),dict(name='Qwen3-VL-Embedding official implementation',url='https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py',use='query-free instruction and last-nonpadding pooling; same local model and processor, no silent truncation')]

def proposals(key):
 q=read(PARENT/'queries'/f'{key}.json');assert q['complete'];trees={m:read(PARENT/'search'/key/f'{m}.json') for m in METHODS};sets={};union=set()
 for method,t in trees.items():
  for budget in BUDGETS:
   s=t['snapshots'][str(budget)];masses=collections.defaultdict(float)
   for leaf in t['leaves'][:s['leaf_count']]:
    if leaf['valid']:masses[tuple(leaf['span'])]+=math.exp(leaf['logp'])
   sets[f'{method}:{budget}']=dict(method=method,budget=budget,masses=masses,snapshot=s);union.update(masses)
 if q['greedy']['span']:union.add(tuple(q['greedy']['span']))
 spans=sorted(union);index={s:i for i,s in enumerate(spans)}
 for c in sets.values():c['members']=[dict(id=index[s],native_mass=v) for s,v in sorted(c.pop('masses').items())]
 return q,trees,spans,sets

def temporal_modes(spans,weights,h=1.):
 if not len(spans):return dict(assignment=[],groups=[],bandwidth=h)
 x=np.asarray(spans,float);w=np.asarray(weights,float);w=w/w.sum();dist=((x[:,None]-x[None,:])**2).sum(-1);density=np.exp(-dist/(2*h*h))@w/(2*np.pi*h*h);n=len(x);parent=list(range(n));peaks=density.copy();edges=set()
 for i in range(n):
  for j in np.argsort(dist[i],kind='stable')[1:min(n,9)]:
   if dist[i,j]<=16*h*h:edges.add((min(i,int(j)),max(i,int(j))))
 edges=sorted(edges);saddles=[]
 for start in range(0,len(edges),128):
  part=edges[start:start+128];mid=np.asarray([(x[a]+x[b])/2 for a,b in part]);d=((mid[:,None]-x[None,:])**2).sum(-1);md=np.exp(-d/(2*h*h))@w/(2*np.pi*h*h)
  saddles.extend((min(float(v),float(density[a]),float(density[b])),a,b) for (a,b),v in zip(part,md))
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for saddle,a,b in sorted(saddles,reverse=True):
  a,b=root(a),root(b)
  if a!=b and saddle/max(1e-300,min(peaks[a],peaks[b]))>=.5:
   if (peaks[b],-b)>(peaks[a],-a):a,b=b,a
   parent[b]=a;peaks[a]=max(peaks[a],peaks[b])
 clusters=collections.defaultdict(list)
 for i in range(n):clusters[root(i)].append(i)
 groups=sorted(clusters.values(),key=lambda ids:(-sum(w[ids]),ids[0]));assignment=[0]*n;out=[]
 for g,ids in enumerate(groups):
  for i in ids:assignment[i]=g
  ids=np.asarray(ids);mass=float(w[ids].sum());representative=int(ids[np.argmax(w[ids])]);out.append(dict(id=g,members=ids.tolist(),weight=mass,representative=representative,span=list(spans[representative]),mean=(w[ids]@x[ids]/mass).tolist()))
 return dict(assignment=assignment,groups=out,bandwidth=h,saddle_ratio=.5,graph_neighbors=8,graph_radius=4*h)

def weighted_stats(values,weights=None):
 a=np.asarray(values,float)
 if not len(a):return dict(n=0,mean=None,std=None,q10=None,median=None,q90=None)
 w=np.ones(len(a),float) if weights is None else np.asarray(weights,float);w=w/w.sum();order=np.argsort(a,kind='stable');c=np.cumsum(w[order]);quant=lambda p:float(a[order[min(len(a)-1,int(np.searchsorted(c,p)))]]);mean=float(w@a)
 return dict(n=len(a),mean=mean,std=float(np.sqrt(w@((a-mean)**2))),q10=quant(.1),median=quant(.5),q90=quant(.9))

def semantic_summary(spans,members,vectors,greedy_index,mode):
 ids=[r['id'] for r in members];weights=np.array([r['native_mass'] for r in members]);weights=weights/weights.sum() if len(weights) else weights;out={}
 if not ids:return out
 a,b=np.triu_indices(len(ids),1);pairs=[(ids[i],ids[j]) for i,j in zip(a,b)];overlap=np.array([iou(spans[i],spans[j]) for i,j in pairs]);pair_w=weights[a]*weights[b];same=np.array([mode['assignment'][ids[i]]==mode['assignment'][ids[j]] for i,j in zip(a,b)])
 for name,v in vectors.items():
  z=np.asarray(v,float);cos=np.clip(z[ids]@z[ids].T,-1,1);values=cos[a,b];stats=dict(all=weighted_stats(values,pair_w) if len(values) else weighted_stats([]),within_mode=weighted_stats(values[same],pair_w[same]) if same.any() else weighted_stats([]),between_modes=weighted_stats(values[~same],pair_w[~same]) if (~same).any() else weighted_stats([]),overlap_bins={})
  for lo,hi in [(0,.1),(.1,.5),(.5,1.000001)]:
   mask=(overlap>=lo)&(overlap<hi);stats['overlap_bins'][f'{lo:g}:{min(1,hi):g}']=weighted_stats(values[mask],pair_w[mask]) if mask.any() else weighted_stats([])
  if greedy_index is not None:stats['to_greedy']=[dict(id=i,cosine=float(np.clip(z[i]@z[greedy_index],-1,1))) for i in ids]
  selected=np.argsort(values,kind='stable')[:min(12,len(values))];stats['most_different_pairs']=[dict(a=pairs[k][0],b=pairs[k][1],cosine=float(values[k]),temporal_iou=float(overlap[k])) for k in selected]
  out[name]=stats
 return out

def branch_statistics(tree,snapshot,spans,modes):
 index={tuple(s):i for i,s in enumerate(spans)};by_node=collections.defaultdict(list)
 for leaf in tree['leaves'][:snapshot['leaf_count']]:
  if not leaf['valid']:continue
  at=leaf['node']
  while at is not None:by_node[at].append(leaf);at=tree['nodes'][at]['parent']
 def stats(leaves):
  atoms,scalar=interval_distribution(leaves);mass=sum(r['native_mass'] for r in atoms);group=collections.defaultdict(float)
  for r in atoms:group[modes['assignment'][index[tuple(r['span'])]]]+=r['native_mass']
  h=-sum(v/mass*math.log2(v/mass) for v in group.values()) if mass else None
  return dict(valid_mass=mass,interval_count=len(atoms),interval_entropy=scalar['entropy'],mean=scalar['mean'],covariance=scalar['covariance'],mode_entropy=h)
 out=[]
 for n in tree['nodes']:
  if not n.get('visited_step') or n['visited_step']>snapshot['logical_step'] or len(n.get('children',[]))<2:continue
  parent=stats(by_node[n['id']]);cs=[dict(node=c,token=tree['nodes'][c]['piece'],native_token_probability=tree['nodes'][c]['p'],**stats(by_node[c])) for c in n['children']];mass=parent['valid_mass'];mi=parent['mode_entropy']-sum(c['valid_mass']/mass*c['mode_entropy'] for c in cs if c['valid_mass']) if mass else None
  out.append(dict(node=n['id'],prefix=n.get('prefix_text'),role=n['role'],local_token_entropy_bits=n.get('entropy'),observed_mode_mi_bits=max(0,mi) if mi is not None else None,**parent,children=cs))
 return out
