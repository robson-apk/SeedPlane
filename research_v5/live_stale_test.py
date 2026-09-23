"""Generate actual old/new-generation model outputs concurrently. No payload permutation."""
import time,sys,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from runtime_benchmark import PROJECT,ROOT,init_worker,infer
from routing import MODES,accept

def main():
 import torch
 sys.path.insert(0,str(PROJECT));import clmp_parity_seed_v3 as v
 rows=[]
 with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  warm=[pool.submit(infer,(0,0,128,0,144,[1]*144,.02)) for _ in range(8)];[f.result() for f in warm]
  for seed in [11,23,37]:
   for rep in range(10):
    rng=np.random.default_rng(seed*100+rep);L=512;a=int(rng.integers(0,len(v.val_ids)-L-1));y=v.val_ids[a:a+L].numpy().copy();mask=rng.random(L)<.6;new=y.copy();new[mask]=1;old=np.full(L,1,dtype=np.int64)
    fs={};submitted=time.perf_counter()
    # Old task already in flight, current generation supersedes it immediately.
    for gen,x,delay in [(0,old,.025),(1,new,0.)]:
     for k,c0,c1,q0,q1 in v.ranges_for(L):fs[pool.submit(infer,(k,c0,c1,q0,q1,x[q0:q1].tolist(),delay))]=(gen,k)
    delivery=[];current={}
    for f in as_completed(fs):
     gen,k=fs[f];r=f.result();delivery.append((gen,k,r))
     if gen==1:current[k]=r
    metrics={}
    for mode in MODES:
     seen=set();false=0;valid=0
     for gen,k,r in delivery:
      e=(seed*100+rep,1,k,1,r['q0'],k);actual=(e[0],gen,*e[2:]);yes=accept(mode,actual,e,seen)
      false+=int(yes and gen==0);valid+=int(yes and gen==1)
     metrics[mode]={'stale_accepted':false,'current_accepted':valid}
    old_diffs=[float(np.abs(r['p']-current[k]['p']).max()) for gen,k,r in delivery if gen==0]
    rows.append({'seed':seed,'rep':rep,'completion_order':[(g,k) for g,k,r in delivery],'distinct_worker_pids':sorted({r['pid'] for g,k,r in delivery}),'seconds':time.perf_counter()-submitted,'old_current_max_probability_difference':max(old_diffs),'metrics':metrics})
   print('live stale',seed,flush=True)
 (ROOT/'live_stale_results.json').write_text(json.dumps({'scope':'Actual different masked inputs, real concurrent old/current inference with generation bump; 25ms controlled delay on old tasks. No word permutation. Not multi-step diffusion training.','rows':rows},indent=2))
if __name__=='__main__':main()
