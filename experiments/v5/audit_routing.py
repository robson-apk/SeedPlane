import json,time,statistics,os,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from routing import *
ROOT=Path(__file__).resolve().parent;RESULTS=ROOT/'results'

def cases(seed,n=2000):
 rng=np.random.default_rng(seed);rows=[]
 for i in range(n):
  e=(i+1,7,int(rng.integers(1,129)),2,int(rng.integers(0,8192)),int(rng.integers(0,32)))
  rows.append(('clean',e,e,True))
  for kind,col,delta in [('foreign',2,1),('modulo_collision',2,32),('stale',1,-1),('request',0,99999),('model_version',3,-1),('target',4,1),('source',5,1)]:
   a=list(e);a[col]+=delta;rows.append((kind,tuple(a),e,False))
  rows.append(('duplicate',e,e,False))
 return rows

def worker(job):
 i,delay=job;time.sleep(delay);return i,os.getpid()
def main():
 allout=[]
 for seed in (11,23,37):
  rows=cases(seed);res={}
  for mode in MODES:
   seen=set();tot={}
   t=time.perf_counter()
   for kind,a,e,gold in rows:
    yes=accept(mode,a,e,seen);r=tot.setdefault(kind,{'n':0,'accepted':0,'errors':0});r['n']+=1;r['accepted']+=int(yes);r['errors']+=int(yes!=gold)
   res[mode]={'cases':tot,'us_per_message':(time.perf_counter()-t)*1e6/len(rows),'metadata_bytes':wire_bytes(mode)}
  allout.append({'seed':seed,'results':res})
 # Actual out-of-order process completion; payload values in this test are synthetic.
 asyncs=[]
 for seed in (11,23,37):
  rng=np.random.default_rng(seed);jobs=[(i,float(rng.uniform(0,.008))) for i in range(80)];start=time.perf_counter()
  with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
   futures=[pool.submit(worker,j) for j in jobs];order=[f.result() for f in as_completed(futures)]
  asyncs.append({'seed':seed,'workers_observed':len({p for i,p in order}),'out_of_order':sum(i!=j for j,(i,p) in enumerate(order)),'seconds_including_startup':time.perf_counter()-start})
 out={'seeds':allout,'real_process_delivery_synthetic_payload':asyncs,'equivalence':'For the fixed Hadamard lookup, complement > 0.5 is exactly boundary modulo 32 equality; exact envelope validation implies the V5 Hadamard check. V5 cannot improve correctness under the same exact contract.','scope':'Injected metadata faults, not natural LLM quality or measured network fault rates.'}
 (RESULTS/'routing_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
