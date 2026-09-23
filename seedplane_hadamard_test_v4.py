import sys,os,json,math,statistics,time,torch
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
import clmp_parity_seed_v3 as v

torch.set_num_threads(5); torch.manual_seed(v.SEED+12345)
ckpt_path = os.path.join(BASE_DIR, 'clmp_parity_ctx1024.pt')
m=v.ParityDenoiser('clmp');m.load_state_dict(torch.load(ckpt_path,map_location='cpu')['state_dict']);m.eval()

def hadamard(n):
    H=torch.ones(1,1)
    while H.shape[0]<n: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H
KEY_DIM=32; H=hadamard(KEY_DIM)/math.sqrt(KEY_DIM)
def key(bid): return H[bid.long()%KEY_DIM]
def comp_score(src_key,owner_key): return (-(src_key*owner_key).sum(-1)).clamp(0,1)

@torch.no_grad()
def collect(x):
 B,L=x.shape;V=v.VOCAB_SIZE
 oa=torch.zeros(B,L,V);ow=torch.zeros(B,L);ha=torch.zeros(B,L,V);hw=torch.zeros(B,L);ba=torch.zeros(B,L);bw=torch.zeros(B,L)
 for k,c0,c1,q0,q1 in v.ranges_for(L):
  p,bid,sign=v.chunk_meta_paired(k,c0,c1,q0,q1,B);z=m(x[:,q0:q1],p,'none',bid,sign);pr=F.softmax(z,-1);cf=pr.max(-1).values;pos=torch.arange(q0,q1)
  for hi,o in enumerate(v.OFFSETS):
   tgt=pos+o;val=(tgt>=0)&(tgt<L);idx=tgt[val].long();src=pr[:,val,hi,:];w=cf[:,val,hi];ss=sign[:,val];bb=bid[:,val].float();hm=(ss<0).float();om=1-hm
   oa.index_add_(1,idx,src*(w*om).unsqueeze(-1));ow.index_add_(1,idx,w*om);ha.index_add_(1,idx,src*(w*hm).unsqueeze(-1));hw.index_add_(1,idx,w*hm);ba.index_add_(1,idx,bb*w*hm);bw.index_add_(1,idx,w*hm)
 return oa/ow.clamp_min(1e-9).unsqueeze(-1),ha/hw.clamp_min(1e-9).unsqueeze(-1),ow,hw,(ba/bw.clamp_min(1e-9)).round().long(),hw>0

def poe(o,h,ow,hw,lam=.25):
 g=(lam*hw/(ow+hw+1e-9)).clamp(0,.6); lp=(1-g.unsqueeze(-1))*o.clamp_min(1e-9).log()+g.unsqueeze(-1)*h.clamp_min(1e-9).log(); return F.softmax(lp,-1)
def batch(L,B,r):
 g=torch.Generator().manual_seed(v.SEED+71000+L*31+r);st=torch.randint(0,len(v.val_ids)-L-1,(B,),generator=g);y=torch.stack([v.val_ids[int(s):int(s)+L] for s in st]);mm=torch.rand(B,L,generator=g)<.60;x=y.clone();x[mm]=v.MASK_ID;return x,y,mm
def bnll(p,y,mm,L):
 lp=-p.gather(-1,y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log(); bm0=torch.zeros(L,dtype=torch.bool)
 for b in range(v.SHARD,L,v.SHARD): bm0[max(0,b-v.HALO):min(L,b+v.HALO)]=True
 bm=mm&bm0.unsqueeze(0); return float(lp[bm].mean())

# key geometry
ids=torch.arange(1,9); K=key(ids); G=K@K.T
geometry={'key_dim':KEY_DIM,'max_wrong_abs_cos_8boundaries':float((G-torch.eye(len(ids))).abs().max()),'correct_complement':float(comp_score(-K,K).mean())}
out={'geometry':geometry,'contexts':{}}
for L,B,reps in [(512,8,12),(1024,4,12)]:
 res={str(n):{'unseeded':[],'seedplane':[]} for n in [.1,.25,.5]}
 t0=time.perf_counter()
 for r in range(reps):
  x,y,mm=batch(L,B,r);o,h,ow,hw,bid,has=collect(x);clean=poe(o,h,ow,hw);base=bnll(clean,y,mm,L)
  # corrupt/stale message from a different boundary + vocab permutation
  perm=torch.randperm(v.VOCAB_SIZE,generator=torch.Generator().manual_seed(80000+r))
  foreign=torch.roll(h,v.SHARD,1)[...,perm]; fbid=torch.roll(bid,v.SHARD,1)
  # borrowed messages carry -K_source; owner expects +K_target
  ck=key(bid); fk=key(fbid); good=comp_score(-ck,ck); bad=comp_score(-fk,ck)
  for n in [.1,.25,.5]:
   mixed=(1-n)*h+n*foreign
   un=poe(o,mixed,ow,hw)
   wc=(1-n)*good;wf=n*bad;den=(wc+wf).clamp_min(1e-9); routed=(h*wc.unsqueeze(-1)+foreign*wf.unsqueeze(-1))/den.unsqueeze(-1);routed=torch.where(has.unsqueeze(-1),routed,h)
   sp=poe(o,routed,ow,hw)
   res[str(n)]['unseeded'].append(bnll(un,y,mm,L)-base);res[str(n)]['seedplane'].append(bnll(sp,y,mm,L)-base)
 def summary(a):return {'mean_boundary_nll_delta':statistics.mean(a),'stdev':statistics.stdev(a),'batches':len(a)}
 z={}
 for n,d in res.items():
  z[n]={'unseeded':summary(d['unseeded']),'seedplane':summary(d['seedplane']),'seedplane_better_batches':sum(s<u for s,u in zip(d['seedplane'],d['unseeded']))}
 out['contexts'][str(L)]=z;out['contexts'][str(L)]['seconds']=time.perf_counter()-t0
 print('L',L,json.dumps(z,indent=2),flush=True)
open(os.path.join(BASE_DIR, 'seedplane_hadamard_test_v4.json'),'w').write(json.dumps(out,indent=2))
print('GEOMETRY',geometry)
