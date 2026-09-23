"""V5b: sharded 4-worker inference (no jitter) vs single unsharded forward. See PROTOCOL_V5b.md."""
import sys,time,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from runtime_benchmark import PROJECT,ROOT,RESULTS,CKPT,init_worker,infer,packets,fuse

def main():
 import torch
 sys.path.insert(0,str(PROJECT/'seedplane'));import clmp_parity_seed_v3 as v
 model=v.ParityDenoiser('clmp');ck=torch.load(CKPT,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
 def glob(x,threads):
  torch.set_num_threads(threads);L=len(x);t=time.perf_counter()
  with torch.inference_mode():
   h=model.hidden(torch.tensor(x,dtype=torch.long)[None,:],torch.arange(L)[None,:],'none');p=torch.softmax(model.logits_from_repr(h),-1)[0].numpy()
  return time.perf_counter()-t,p
 rows=[]
 with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  [f.result() for f in [pool.submit(infer,(0,0,128,0,144,[1]*144,.02)) for _ in range(8)]]
  for L in [512,1024]:
   for th in [1,4]:
    for _ in range(5):glob([1]*L,th)
   for seed in [11,23,37]:
    for rep in range(20):
     rng=np.random.default_rng(seed*1000+L+rep);a=int(rng.integers(0,len(v.val_ids)-L-1));y=v.val_ids[a:a+L].numpy().copy();x=y.copy();x[rng.random(L)<.6]=1
     modes=['global_1t','global_4t','sharded_4w'];rng.shuffle(modes);t_={}
     for m in modes:
      if m=='sharded_4w':
       jobs=[(k,c0,c1,q0,q1,x[q0:q1].tolist(),0.) for k,c0,c1,q0,q1 in v.ranges_for(L)]
       t=time.perf_counter();res=[f.result() for f in as_completed([pool.submit(infer,j) for j in jobs])];owners,frames=packets(res,seed,rep);fuse('exact_envelope',owners,frames,'clean',L,seed+rep);t_[m]=time.perf_counter()-t
      else:t_[m]=glob(x.tolist(),int(m[-2]))[0]
     rows.append({'L':L,'seed':seed,'rep':rep,'order':modes,'seconds':t_})
    print('v5b',L,seed,flush=True)
 comp=[]
 for L in [512,1024]:
  for seed in [11,23,37]:
   r=[x for x in rows if x['L']==L and x['seed']==seed];s=np.array([x['seconds']['sharded_4w'] for x in r])
   g={k:np.array([x['seconds'][k] for x in r]) for k in ['global_1t','global_4t']};best=min(g,key=lambda k:g[k].mean());b=g[best]
   ix=np.random.default_rng(seed).integers(len(s),size=(2000,len(s)));red=1-s.mean()/b.mean();ci=np.quantile(1-s[ix].mean(1)/b[ix].mean(1),[.025,.975])
   comp.append({'L':L,'seed':seed,'n_pairs':len(s),'median_ms':{k:float(np.median(v_)*1000) for k,v_ in {**g,'sharded_4w':s}.items()},'best_global':best,'reduction_vs_best_global':float(red),'ci95':ci.tolist(),'passes_gate':bool(red>=.10 and ci[0]>0)})
 verdict='SPEEDUP_SUPPORTED' if all(c['passes_gate'] for c in comp if c['L']==1024) else 'SPEEDUP_CLAIM_FALSIFIED'
 (RESULTS/'global_vs_sharded_results.json').write_text(json.dumps({'protocol':'PROTOCOL_V5b.md','scope':'Local CPU (Mac), toy checkpoint, timing only, no jitter.','verdict':verdict,'comparisons':comp,'rows':rows},indent=2))
 print(json.dumps({'verdict':verdict,'comparisons':comp},indent=1))
if __name__=='__main__':main()
