"""Deterministic admitted-path distributions. No sampling or search budget."""
import collections,datetime,hashlib,json,math,os,sqlite3,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'_lib').is_dir() and (p/'artifacts').is_dir())
sys.path.insert(0,str(ROOT.parent/'_shared/timelens/runtime/interval_distribution_deps'))
OUT=ROOT/'artifacts/runs/rollout/interval_distribution/pilot/900'
RESULT=ROOT/'results/rollout/interval_distribution/pilot/900'
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
MODEL=Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct')
EMBED=Path('/home/irteam/backbones/models/Qwen3-VL-Embedding-8B')
VERIFIER=Path('/home/irteam/backbones/models/Qwen3-VL-4B-Instruct')
CONFIG=dict(family='rollout/interval_distribution',scope='pilot',queries=900,top_p=.99,absolute_floor=.01,
 sampling=False,search_budget=None,path_length_budget=None,temperature=1.,input_order='video_query',fps=2,
 spatial_target=100352,entropy_bins='exact parsed clipped endpoint pair; no additional binning',
 path_mass='native full-vocabulary probability product, no step renormalization',
 grouping=dict(grid_seconds=.5,bandwidth_seconds=1.,saddle_ratio=.5,sensitivity=[.5,2.]),
 semantics=['instruct_mean','instruct_temporal_four','instruct_readout','embedding_readout'],
 semantic_instruction="Represent the user's input.",captioning=False,query_in_semantics=False,
 permanent_vectors=False,semantic_feedback=False,worker_count=1,
 verifier=str(VERIFIER),verification_context_seconds=1.,verification=['event_only','event_onset_offset'],
 stages=[300,900],stage_gate='all inference, independent verification, reports, browser and raw validation before extension',
 final_selection='highest pointwise verifier score; baseline included; ties retain baseline then native mass; no new timestamp')
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def logadd(a,b):return float(np.logaddexp(a,b))
def lsum(xs):
 xs=list(xs)
 if not xs:return -math.inf
 m=max(xs);return m+math.log(sum(math.exp(x-m) for x in xs))
def admitted(probabilities):
 p=np.asarray(probabilities,dtype=np.float64);eligible=np.flatnonzero(p>=CONFIG['absolute_floor']);order=eligible[np.lexsort((eligible,-p[eligible]))];edges=[];cum=0.
 for token in order:
  value=float(p[token])
  if cum>=CONFIG['top_p']:break
  cum+=value
  if value<CONFIG['absolute_floor']:break
  edges.append(dict(token=int(token),p=value))
 positive=p[p>0];return edges,float(-(positive*np.log2(positive)).sum())
def connect(path):
 db=sqlite3.connect(path,timeout=60);db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=NORMAL')
 db.executescript('''CREATE TABLE IF NOT EXISTS nodes(id INTEGER PRIMARY KEY,parent INTEGER,token INTEGER,p REAL,logp REAL,depth INTEGER,status TEXT,entropy REAL,accepted TEXT,prefix_text TEXT,span TEXT,raw TEXT,reason TEXT,text TEXT);
 CREATE INDEX IF NOT EXISTS parent_idx ON nodes(parent); CREATE INDEX IF NOT EXISTS pending_idx ON nodes(status,id);
 CREATE TABLE IF NOT EXISTS stats(node INTEGER PRIMARY KEY,data TEXT);
 CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);''');return db

def distribution_stats(masses):
 if not masses:return dict(log_valid_mass=None,entropy=None,mean=None,covariance=None,variance_sum=None,intervals=0)
 z=lsum(masses.values());keys=list(masses);w=np.array([math.exp(masses[k]-z) for k in keys]);x=np.asarray(keys,dtype=float);mu=w@x;delta=x-mu;cov=(delta*w[:,None]).T@delta
 return dict(log_valid_mass=z,entropy=float(-(w*np.log2(np.maximum(w,1e-300))).sum()),mean=mu.tolist(),covariance=cov.tolist(),variance_sum=float(np.trace(cov)),intervals=len(keys))

def summarize_tree(path,emit=True):
 """Bottom-up native suffix log masses. Child weights condition on valid survival."""
 db=connect(path);pending={};child_stats=collections.defaultdict(list);root_masses={};nodes=0
 for row in db.execute('SELECT id,parent,token,p,status,entropy,accepted,span FROM nodes ORDER BY id DESC'):
  ident,parent,token,p,status,entropy,accepted,span=row
  masses=pending.pop(ident,{})
  if status=='valid':masses={tuple(json.loads(span)):0.}
  stat=distribution_stats(masses);children=child_stats.pop(ident,[])
  if children and stat['log_valid_mass'] is not None:
   ws=[math.exp(math.log(c['p'])+c['log_valid_mass']-stat['log_valid_mass']) if c['log_valid_mass'] is not None else 0. for c in children]
   hs=[c['entropy'] or 0. for c in children];mean=sum(w*h for w,h in zip(ws,hs));stat.update(expected_residual_interval_entropy=mean,residual_interval_entropy_variance=sum(w*(h-mean)**2 for w,h in zip(ws,hs)),interval_information_bits=max(0.,stat['entropy']-mean))
   mus=[np.asarray(c['mean']) if c['mean'] else np.zeros(2) for c in children];mu=np.asarray(stat['mean']);between=sum((w*np.outer(m-mu,m-mu) for w,m in zip(ws,mus)),np.zeros((2,2)));within=sum((w*np.asarray(c['covariance']) if c['covariance'] else np.zeros((2,2)) for w,c in zip(ws,children)),np.zeros((2,2)));stat.update(within_branch_covariance=within.tolist(),between_branch_covariance=between.tolist())
  edges=json.loads(accepted) if accepted else []
  if children:
   ps=sum(c['p'] for c in children);mean=sum(c['p']*(c['next_entropy'] or 0.) for c in children)/ps if ps else 0.
   stat.update(expected_child_token_entropy=mean,child_token_entropy_variance=sum(c['p']/ps*((c['next_entropy'] or 0.)-mean)**2 for c in children) if ps else None)
  if emit:db.execute('INSERT OR REPLACE INTO stats VALUES (?,?)',(ident,json.dumps(stat,allow_nan=False)))
  if parent is not None:
   target=pending.setdefault(parent,{})
   for k,v in masses.items():target[k]=logadd(target.get(k,-math.inf),math.log(p)+v)
   child_stats[parent].append(dict(p=p,next_entropy=entropy,**stat))
  else:root_masses=masses;root_stat=stat
  nodes+=1
 db.commit();terminals=collections.Counter();mass=collections.defaultdict(list)
 for status,logp,accepted in db.execute('SELECT status,logp,accepted FROM nodes'):
  terminals[status]+=1
  if status in ('valid','invalid','dead','context_limit'):mass[status].append(logp)
  if status=='expanded':
   retained=sum(e['p'] for e in json.loads(accepted));excluded=max(0.,1-retained)
   if excluded:mass['filtered'].append(logp+math.log(excluded))
 summary=dict(nodes=nodes,counts=dict(terminals),root=root_stat,complete=not any(terminals[x] for x in ('pending','context_limit')),mass={k:math.exp(lsum(v)) for k,v in mass.items()})
 total=summary['root']['log_valid_mass'];intervals=[dict(id=digest(list(k))[:20],span=list(k),log_native_mass=v,conditional_weight=math.exp(v-total)) for k,v in sorted(root_masses.items())]
 db.close();return summary,intervals

def partition(intervals,bandwidth=1.):
 """2-D density watershed; merge shallow saddles. Does not modify entropy atoms."""
 if not intervals:return [],{}
 from scipy.ndimage import gaussian_filter,maximum_filter,label
 from skimage.segmentation import watershed
 x=np.array([r['span'] for r in intervals]);weights=np.array([r['conditional_weight'] for r in intervals]);delta=CONFIG['grouping']['grid_seconds'];origin=np.floor(x.min(axis=0)/delta)*delta-3*bandwidth;idx=np.floor((x-origin)/delta).astype(int);shape=tuple((idx.max(axis=0)+1+math.ceil(3*bandwidth/delta)).tolist());hist=np.zeros(shape,dtype=float);np.add.at(hist,(idx[:,0],idx[:,1]),weights)
 density=gaussian_filter(hist,bandwidth/delta);ss=origin[0]+(np.arange(shape[0])+.5)*delta;ee=origin[1]+(np.arange(shape[1])+.5)*delta;mask=(ss[:,None]<ee[None,:]+delta)&(density>0);peaks=(density==maximum_filter(density,size=3))&mask;markers,n=label(peaks,np.ones((3,3)));basins=watershed(-density,markers,mask=mask);heights={i:float(density[markers==i].max()) for i in range(1,n+1)};parent={i:i for i in heights}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 saddles={}
 for axis in (0,1):
  a=basins[:-1,:] if axis==0 else basins[:,:-1];b=basins[1:,:] if axis==0 else basins[:,1:];da=density[:-1,:] if axis==0 else density[:,:-1];dB=density[1:,:] if axis==0 else density[:,1:];valid=(a!=b)&(a>0)&(b>0)
  for u,v,h in zip(a[valid],b[valid],np.minimum(da,dB)[valid]):
   key=tuple(sorted((int(u),int(v))));saddles[key]=max(saddles.get(key,0.),float(h))
 for (u,v),height in sorted(saddles.items(),key=lambda pair:-pair[1]):
  u,v=find(u),find(v)
  if u!=v and height>CONFIG['grouping']['saddle_ratio']*min(heights[u],heights[v]):
   keep,drop=sorted((u,v),key=lambda i:(-heights[i],i));parent[drop]=keep;heights[keep]=max(heights[keep],heights[drop])
 assignment=[find(int(basins[i,j])) for i,j in idx];ids={k:i for i,k in enumerate(sorted(set(assignment),key=lambda k:(-sum(w for a,w in zip(assignment,weights) if a==k),k)))};assignments={r['id']:ids[a] for r,a in zip(intervals,assignment)};groups=[]
 for group in sorted(ids.values()):
  members=[r for r in intervals if assignments[r['id']]==group];groups.append(dict(id=group,interval_ids=[r['id'] for r in members],mass=sum(r['conditional_weight'] for r in members)))
 return groups,assignments

def grouped(intervals):
 groups,assign=partition(intervals)
 other=[partition(intervals,h)[1] for h in CONFIG['grouping']['sensitivity']]
 for g in groups:
  stable=True
  for mapping in other:
   labels={mapping[k] for k in g['interval_ids']}
   if len(labels)!=1 or {k for k,v in mapping.items() if v in labels}!=set(g['interval_ids']):stable=False
  g['scale_stable']=stable
 return groups,assign
