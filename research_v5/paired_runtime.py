"""Independent paired end-to-end calls with randomized router order, persistent workers."""
import sys,time,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from runtime_benchmark import PROJECT,ROOT,init_worker,infer,packets,fuse

def main():
 import torch
 sys.path.insert(0,str(PROJECT));import clmp_parity_seed_v3 as v
 torch.set_num_threads(1);rows=[]
 for nw in [1,2,4]:
  with ProcessPoolExecutor(max_workers=nw,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
   warm=[pool.submit(infer,(0,0,128,0,144,[1]*144,.02)) for _ in range(nw*2)];[f.result() for f in warm]
   for seed in [11,23,37]:
    for rep in range(20):
     L=1024;rng=np.random.default_rng(seed*1000+rep);a=int(rng.integers(0,len(v.val_ids)-L-1));y=v.val_ids[a:a+L].numpy().copy();mask=rng.random(L)<.6;x=y.copy();x[mask]=1
     modes=['exact_envelope','seedplane_v5'];rng.shuffle(modes);jobs=[(k,c0,c1,q0,q1,x[q0:q1].tolist(),float(rng.uniform(0,.005))) for k,c0,c1,q0,q1 in v.ranges_for(L)];values={};outputs={}
     for mode in modes:
      t=time.perf_counter();fs=[pool.submit(infer,j) for j in jobs];res=[f.result() for f in as_completed(fs)];owners,frames=packets(res,seed,rep);out,route,accepted=fuse(mode,owners,frames,'stale',L,seed+rep);values[mode]={'seconds':time.perf_counter()-t,'route_seconds':route};outputs[mode]=out
     rows.append({'workers':nw,'seed':seed,'rep':rep,'timings':values,'max_abs_output_difference':float(np.abs(outputs['exact_envelope']-outputs['seedplane_v5']).max())})
    print('paired',nw,seed,flush=True)
   (ROOT/'paired_runtime_results.json').write_text(json.dumps({'scope':'Actual independent inference+IPC+fusion calls; same inputs/jitter within pair; randomized order. Local CPU only. Injected stale metadata and permuted payload, not real observed network corruption.','rows':rows},indent=2))
if __name__=='__main__':main()
