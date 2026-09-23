import sys,os,math,json,torch
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(REPO_DIR, 'seedplane'))
import clmp_parity_seed_v3 as v

torch.set_num_threads(5)
torch.manual_seed(v.SEED+12345)
ckpt_path = os.path.join(REPO_DIR, 'checkpoints', 'clmp_parity_ctx1024.pt')
m=v.ParityDenoiser('clmp');m.load_state_dict(torch.load(ckpt_path,map_location='cpu')['state_dict']);m.eval()

def base_seed(pos,bid):
    half=v.D//2;j=torch.arange(half,dtype=torch.float32).view(1,1,-1)
    p=pos.float().unsqueeze(-1);b=bid.float().unsqueeze(-1)
    freq=torch.exp(-math.log(10000.)*j/max(1,half-1));phase=(p+17.*b+.5)*freq
    return F.normalize(torch.cat([torch.sin(phase),torch.cos(phase)],-1)[...,:v.D],dim=-1)

@torch.no_grad()
def collect(x):
    B,L=x.shape;V=v.VOCAB_SIZE
    all_a=torch.zeros(B,L,V); all_w=torch.zeros(B,L)
    own_a=torch.zeros(B,L,V); own_w=torch.zeros(B,L)
    halo_a=torch.zeros(B,L,V);halo_w=torch.zeros(B,L)
    comp_a=torch.zeros(B,L);comp_w=torch.zeros(B,L)
    for k,c0,c1,q0,q1 in v.ranges_for(L):
        p,bid,sign=v.chunk_meta_paired(k,c0,c1,q0,q1,B)
        z=m(x[:,q0:q1],p,'none',bid,sign);pr=F.softmax(z,-1);cf=pr.max(-1).values
        pos=torch.arange(q0,q1); signed=base_seed(p,bid)*sign.float().unsqueeze(-1)
        for hi,o in enumerate(v.OFFSETS):
            tgt=pos+o;valid=(tgt>=0)&(tgt<L);idx=tgt[valid].long()
            src=pr[:,valid,hi,:];w=cf[:,valid,hi]
            all_a.index_add_(1,idx,src*w.unsqueeze(-1));all_w.index_add_(1,idx,w)
            ss=sign[:,valid]; bb=bid[:,valid]; sv=signed[:,valid]
            tp=tgt[valid].unsqueeze(0).expand(B,-1); owner=base_seed(tp,bb)
            compat=(-(sv*owner).sum(-1)).clamp(0,1) # only true complementary -S gets positive
            # proposals from borrowed halo form the seed-complement channel; all others owner/base channel
            hm=(ss<0).float(); om=1.-hm
            halo_a.index_add_(1,idx,src*(w*hm).unsqueeze(-1));halo_w.index_add_(1,idx,w*hm)
            own_a.index_add_(1,idx,src*(w*om).unsqueeze(-1));own_w.index_add_(1,idx,w*om)
            comp_a.index_add_(1,idx,compat*w*hm);comp_w.index_add_(1,idx,w*hm)
    standard=all_a/all_w.clamp_min(1e-9).unsqueeze(-1)
    owner=own_a/own_w.clamp_min(1e-9).unsqueeze(-1)
    halo=halo_a/halo_w.clamp_min(1e-9).unsqueeze(-1)
    comp=comp_a/comp_w.clamp_min(1e-9)
    has=halo_w>0
    return standard,owner,halo,comp,has,own_w,halo_w

def fuse(owner,halo,comp,has,own_w,halo_w,mode,param):
    if mode=='mix':
        # seed compatibility determines max halo contribution; confidence mass chooses relative importance
        ratio=halo_w/(halo_w+own_w+1e-9)
        g=(float(param)*comp*ratio).clamp(0,.8);g=torch.where(has,g,torch.zeros_like(g))
        return owner*(1-g.unsqueeze(-1))+halo*g.unsqueeze(-1)
    if mode=='poe':
        ratio=halo_w/(halo_w+own_w+1e-9);g=(float(param)*comp*ratio).clamp(0,.6);g=torch.where(has,g,torch.zeros_like(g))
        logp=(1-g.unsqueeze(-1))*owner.clamp_min(1e-9).log()+g.unsqueeze(-1)*halo.clamp_min(1e-9).log()
        return F.softmax(logp,-1)
    if mode=='agree':
        # only use complement when distributions agree; otherwise trust owner/base evidence
        agree=torch.sqrt((owner*halo).clamp_min(0)).sum(-1) # Bhattacharyya coefficient
        ratio=halo_w/(halo_w+own_w+1e-9)
        g=(float(param)*comp*ratio*agree).clamp(0,.7);g=torch.where(has,g,torch.zeros_like(g))
        return owner*(1-g.unsqueeze(-1))+halo*g.unsqueeze(-1)
    raise ValueError(mode)

def batch(L,B,rep):
    g=torch.Generator().manual_seed(v.SEED+41000+L*31+rep)
    starts=torch.randint(0,len(v.val_ids)-L-1,(B,),generator=g)
    y=torch.stack([v.val_ids[int(s):int(s)+L] for s in starts]);mm=torch.rand(B,L,generator=g)<.60;x=y.clone();x[mm]=v.MASK_ID
    return x,y,mm

def newS():return {'nll':0.,'bnll':0.,'inll':0.,'n':0,'bn':0,'inn':0,'ok':0,'bok':0}
def add(S,p,y,mm,L):
    lp=-p.gather(-1,y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log();pred=p.argmax(-1)
    bm0=torch.zeros(L,dtype=torch.bool)
    for b in range(v.SHARD,L,v.SHARD):bm0[max(0,b-v.HALO):min(L,b+v.HALO)]=True
    bm=mm&bm0.unsqueeze(0);im=mm&~bm0.unsqueeze(0)
    S['nll']+=float(lp[mm].sum());S['bnll']+=float(lp[bm].sum());S['inll']+=float(lp[im].sum());S['n']+=int(mm.sum());S['bn']+=int(bm.sum());S['inn']+=int(im.sum());S['ok']+=int((pred.eq(y)&mm).sum());S['bok']+=int((pred.eq(y)&bm).sum())
def fin(S):return {'nll':S['nll']/S['n'],'boundary_nll':S['bnll']/S['bn'],'interior_nll':S['inll']/S['inn'],'acc':S['ok']/S['n'],'boundary_acc':S['bok']/S['bn'],'n':S['n'],'bn':S['bn']}

settings=[('standard',0),('mix',.5),('mix',1.),('mix',2.),('poe',.25),('poe',.5),('poe',1.),('agree',1.),('agree',2.)]
out={}
for L,B,reps in [(256,16,5),(512,8,5),(1024,4,5)]:
    ss={f'{mode}_{p}':newS() for mode,p in settings};meta={'comp_sum':0.,'comp_n':0,'halo_positions':0,'total_positions':0}
    for r in range(reps):
        x,y,mm=batch(L,B,r);standard,owner,halo,comp,has,ow,hw=collect(x)
        for mode,p in settings:
            pred=standard if mode=='standard' else fuse(owner,halo,comp,has,ow,hw,mode,p)
            add(ss[f'{mode}_{p}'],pred,y,mm,L)
        meta['comp_sum']+=float(comp[has].sum());meta['comp_n']+=int(has.sum());meta['halo_positions']+=int(has.sum());meta['total_positions']+=has.numel()
    out[str(L)]={k:fin(S) for k,S in ss.items()};out[str(L)]['meta']={'mean_complement_score':meta['comp_sum']/max(1,meta['comp_n']),'halo_position_fraction':meta['halo_positions']/meta['total_positions']}
    print('L',L)
    for k,z in out[str(L)].items():
        if k!='meta':print(k,z)
    print('meta',out[str(L)]['meta'],flush=True)
open(os.path.join(BASE_DIR, 'results', 'seed_fusion_ablation_v4.json'),'w').write(json.dumps(out,indent=2))
