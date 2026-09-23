import sys,os,math,json,time,statistics,torch
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
import clmp_parity_seed_v3 as v

torch.set_num_threads(5);torch.manual_seed(v.SEED+12345)
ckpt_path = os.path.join(BASE_DIR, 'clmp_parity_ctx1024.pt')
m=v.ParityDenoiser('clmp');m.load_state_dict(torch.load(ckpt_path,map_location='cpu')['state_dict']);m.eval()

def bs(pos,bid):
 h=v.D//2;j=torch.arange(h).float().view(1,1,-1);p=pos.float().unsqueeze(-1);b=bid.float().unsqueeze(-1);f=torch.exp(-math.log(10000.)*j/max(1,h-1));ph=(p+17*b+.5)*f
 return F.normalize(torch.cat([torch.sin(ph),torch.cos(ph)],-1)[...,:v.D],dim=-1)
@torch.no_grad()
def collect(x):
 B,L=x.shape;V=v.VOCAB_SIZE; aa=torch.zeros(B,L,V);aw=torch.zeros(B,L);oa=torch.zeros_like(aa);ow=torch.zeros_like(aw);ha=torch.zeros_like(aa);hw=torch.zeros_like(aw);ca=torch.zeros_like(aw);cw=torch.zeros_like(aw)
 for k,c0,c1,q0,q1 in v.ranges_for(L):
  p,bid,sign=v.chunk_meta_paired(k,c0,c1,q0,q1,B);z=m(x[:,q0:q1],p,'none',bid,sign);pr=F.softmax(z,-1);cf=pr.max(-1).values;pos=torch.arange(q0,q1);signed=bs(p,bid)*sign.float().unsqueeze(-1)
  for hi,o in enumerate(v.OFFSETS):
   tgt=pos+o;val=(tgt>=0)&(tgt<L);idx=tgt[val].long();src=pr[:,val,hi,:];w=cf[:,val,hi];ss=sign[:,val];bb=bid[:,val];sv=signed[:,val];tp=tgt[val].unsqueeze(0).expand(B,-1);ownseed=bs(tp,bb);comp=(-(sv*ownseed).sum(-1)).clamp(0,1);hm=(ss<0).float();om=1-hm
   aa.index_add_(1,idx,src*w.unsqueeze(-1));aw.index_add_(1,idx,w);oa.index_add_(1,idx,src*(w*om).unsqueeze(-1));ow.index_add_(1,idx,w*om);ha.index_add_(1,idx,src*(w*hm).unsqueeze(-1));hw.index_add_(1,idx,w*hm);ca.index_add_(1,idx,comp*w*hm);cw.index_add_(1,idx,w*hm)
 return aa/aw.clamp_min(1e-9).unsqueeze(-1),oa/ow.clamp_min(1e-9).unsqueeze(-1),ha/hw.clamp_min(1e-9).unsqueeze(-1),ca/cw.clamp_min(1e-9),hw>0,ow,hw

def poe(owner,halo,comp,has,ow,hw,lam=.25):
 ratio=hw/(ow+hw+1e-9);g=(lam*comp*ratio).clamp(0,.6);g=torch.where(has,g,torch.zeros_like(g));lp=(1-g.unsqueeze(-1))*owner.clamp_min(1e-9).log()+g.unsqueeze(-1)*halo.clamp_min(1e-9).log();return F.softmax(lp,-1)
def metrics(p,y,mm,L):
 lp=-p.gather(-1,y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log();bm0=torch.zeros(L,dtype=torch.bool)
 for b in range(v.SHARD,L,v.SHARD):bm0[max(0,b-v.HALO):min(L,b+v.HALO)]=True
 bm=mm&bm0.unsqueeze(0);return float(lp[mm].mean()),float(lp[bm].mean())
def batch(L,B,r):
 g=torch.Generator().manual_seed(v.SEED+51000+L*31+r);st=torch.randint(0,len(v.val_ids)-L-1,(B,),generator=g);y=torch.stack([v.val_ids[int(s):int(s)+L] for s in st]);mm=torch.rand(B,L,generator=g)<.60;x=y.clone();x[mm]=v.MASK_ID;return x,y,mm
out={}
for L,B,reps in [(512,8,20),(1024,4,20)]:
 ds={'seed_total':[],'seed_boundary':[],'const_total':[],'const_boundary':[],'random_total':[],'random_boundary':[]}; compvals=[]
 t=time.perf_counter()
 for r in range(reps):
  x,y,mm=batch(L,B,r);std,o,h,c,has,ow,hw=collect(x);base=metrics(std,y,mm,L)
  variants={'seed':c,'const':torch.ones_like(c),'random':torch.rand(c.shape,generator=torch.Generator().manual_seed(9000+r))*has.float()}
  compvals.extend(c[has].flatten().tolist())
  for name,cc in variants.items():
   q=metrics(poe(o,h,cc,has,ow,hw),y,mm,L);ds[name+'_total'].append(q[0]-base[0]);ds[name+'_boundary'].append(q[1]-base[1])
 def summ(a):
  return {'mean_delta':statistics.mean(a),'stdev':statistics.stdev(a),'sem':statistics.stdev(a)/(len(a)**.5),'wins':sum(x<0 for x in a),'n':len(a)}
 out[str(L)]={k:summ(a) for k,a in ds.items()};out[str(L)]['seed_score']={'mean':statistics.mean(compvals),'stdev':statistics.stdev(compvals),'min':min(compvals),'max':max(compvals)};out[str(L)]['seconds']=time.perf_counter()-t
 print('L',L,json.dumps(out[str(L)],indent=2),flush=True)
open(os.path.join(BASE_DIR, 'seed_router_robust_v4.json'),'w').write(json.dumps(out,indent=2))
