"""Frozen toy inference on persistent processes; center-head halo fusion only.
No claim of multi-step diffusion, network clusters, or original five-head fusion parity.
"""
import os,sys,time,json,resource,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from routing import accept,MODES,wire_bytes
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent;RESULTS=ROOT/'results';CKPT=PROJECT/'checkpoints'/'clmp_parity_ctx1024.pt'
MODEL=None;V=None

def init_worker():
 global MODEL,V
 sys.path.insert(0,str(PROJECT/'seedplane'));import clmp_parity_seed_v3 as v;import torch
 V=v;torch.set_num_threads(1);MODEL=v.ParityDenoiser('clmp');ck=torch.load(CKPT,map_location='cpu',weights_only=True);MODEL.load_state_dict(ck['state_dict']);MODEL.eval()
def infer(job):
 import torch
 idx,c0,c1,q0,q1,words,delay=job
 start=time.perf_counter()
 with torch.inference_mode():
  x=torch.tensor(words,dtype=torch.long)[None,:];pos=torch.arange(q0,q1)[None,:]
  h=MODEL.hidden(x,pos,'none');p=torch.softmax(MODEL.logits_from_repr(h),-1)[0].numpy()
 compute=time.perf_counter()-start
 time.sleep(delay)
 return {'idx':idx,'c0':c0,'c1':c1,'q0':q0,'q1':q1,'p':p,'pid':os.getpid(),'compute_s':compute,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def packets(results,request,gen):
 frames=[];owners=[]
 for r in results:
  c0,c1,q0,q1=r['c0'],r['c1'],r['q0'],r['q1'];idx=r['idx'];p=r['p'];owners.append((c0,c1,p[c0-q0:c1-q0]))
  if q0<c0:
   e=(request,gen,c0//128,1,q0,idx);frames.append((e,p[:c0-q0]))
  if q1>c1:
   e=(request,gen,c1//128,1,c1,idx);frames.append((e,p[c1-q0:]))
 return owners,frames

def fuse(mode,owners,frames,scenario,L,seed):
 # Transport dispatch slot supplies expected envelope; all modes see identical frames.
 rng=np.random.default_rng(seed);stream=[]
 for e,p in frames:
  stream.append((e,e,p))
  if scenario=='clean':continue
  a=list(e)
  if scenario=='foreign':a[2]+=1
  elif scenario=='collision':a[2]+=32
  elif scenario=='stale':a[1]-=1
  elif scenario=='request':a[0]+=1000
  elif scenario=='duplicate':
   stream.append((e,e,p));continue
  else:raise ValueError(scenario)
  stream.append((tuple(a),e,p[:,rng.permutation(p.shape[1])]))
 rng.shuffle(stream)
 # Owners always exist; halo contributions are averaged with equal nominal weight.
 accum=np.zeros((L,1024),np.float64);count=np.ones(L)
 for a,b,p in owners:accum[a:b]=p
 seen=set();route_s=0.;accepted=0
 for actual,expected,p in stream:
  t=time.perf_counter();yes=accept(mode,actual,expected,seen);route_s+=time.perf_counter()-t
  if yes:
   a=expected[4];b=a+len(p);accum[a:b]+=.25*p;count[a:b]+=.25;accepted+=1
 return accum/count[:,None],route_s,accepted

def main():
 import torch
 sys.path.insert(0,str(PROJECT/'seedplane'));import clmp_parity_seed_v3 as v
 torch.set_num_threads(1)
 checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
 rows=[];qualities=[];pids=[];serials=[]
 for nw in [1,2,4]:
  start=time.perf_counter()
  with ProcessPoolExecutor(max_workers=nw,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
   # Warm all workers with real inference and account initialization separately.
   warm=[pool.submit(infer,(0,0,128,0,144,[1]*144,.02)) for _ in range(nw*2)]
   [f.result() for f in warm];startup=time.perf_counter()-start
   for seed in [11,23,37]:
    for L in [512,1024]:
     for rep in range(6):
      rng=np.random.default_rng(seed*10000+L+rep);a=int(rng.integers(0,len(v.val_ids)-L-1));y=v.val_ids[a:a+L].numpy().copy();mask=rng.random(L)<.6;x=y.copy();x[mask]=1
      # Zero delay and injected worker jitter are measured separately.
      jitter=rep>=3
      jobs=[(k,c0,c1,q0,q1,x[q0:q1].tolist(),float(rng.uniform(0,.008)) if jitter else 0.) for k,c0,c1,q0,q1 in v.ranges_for(L)]
      t=time.perf_counter();fs=[pool.submit(infer,j) for j in jobs];res=[f.result() for f in as_completed(fs)];elapsed=time.perf_counter()-t
      owners,frames=packets(res,seed,rep);clean,_,_=fuse('exact_envelope',owners,frames,'clean',L,seed+rep)
      bn=np.zeros(L,bool)
      for b in range(128,L,128):bn[max(0,b-16):min(L,b+16)]=True
      selected=mask&bn;loss=lambda p:float(-np.log(np.maximum(p[np.arange(L),y],1e-9))[selected].mean())
      modes=list(MODES);rng.shuffle(modes)
      for mode in modes:
       for scenario in ['clean','foreign','collision','stale','request','duplicate']:
        t=time.perf_counter();out,route_s,accepted=fuse(mode,owners,frames,scenario,L,seed+rep);fusion_s=time.perf_counter()-t
        qualities.append({'workers':nw,'seed':seed,'L':L,'rep':rep,'scenario':scenario,'mode':mode,'boundary_nll':loss(out),'delta_vs_clean':loss(out)-loss(clean),'max_abs_vs_clean':float(np.abs(out-clean).max()),'route_s':route_s,'fusion_s':fusion_s,'total_s_shared_inference':elapsed+fusion_s,'accepted':accepted})
      rows.append({'workers':nw,'seed':seed,'L':L,'rep':rep,'jitter':jitter,'inference_s':elapsed,'startup_s':startup,'worker_pids':sorted({r['pid'] for r in res}),'out_of_order':sum(r['idx']!=i for i,r in enumerate(res)),'returned_payload_bytes':sum(r['p'].nbytes for r in res),'max_worker_peak_rss_bytes':max(r['rss_bytes'] for r in res),'sum_task_compute_s':sum(r['compute_s'] for r in res)})
    print('runtime',nw,seed,'done',flush=True)
  (RESULTS/'runtime_results.json').write_text(json.dumps({'scope':'Persistent local CPU processes, frozen toy checkpoint, center-head owner/halo fusion. Scheduler seeds only. Model evaluation noise is synthetic wrong-vocabulary payload; no natural quality gain claimed.','timing_note':'Inference shared across router variants to isolate routing; total_s_shared_inference is inference plus separately timed fusion, not independent end-to-end runs. RSS is peak per worker, not summed physical memory.','rows':rows,'quality_rows':qualities},indent=2))
if __name__=='__main__':main()
