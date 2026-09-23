import os, sys, json, math, time, functools
import torch
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
import clmp_parity_seed_v3 as v

# Seed is no longer added to hidden states. It becomes a parameter-free routing field.
# Same global overlap token gets +S in owner/core and -S in borrowed halo.
# Attention gets a positive bias only for complementary (+/-) pairs on the same boundary.

def base_seed(pos,bid):
    half=v.D//2
    j=torch.arange(half,device=pos.device,dtype=torch.float32).view(1,1,-1)
    p=pos.float().unsqueeze(-1); b=bid.float().unsqueeze(-1)
    freq=torch.exp(-math.log(10000.0)*j/max(1,half-1))
    phase=(p+17.0*b+0.5)*freq
    s=torch.cat([torch.sin(phase),torch.cos(phase)],dim=-1)
    if s.size(-1)<v.D: s=F.pad(s,(0,v.D-s.size(-1)))
    return F.normalize(s[...,:v.D],dim=-1)

@functools.lru_cache(maxsize=128)
def cached_attn_bias(k,c0,c1,q0,q1,strength):
    p,bid,sign=v.chunk_meta_paired(k,c0,c1,q0,q1,1)
    signed=base_seed(p,bid)[0] * sign[0].float().unsqueeze(-1)
    # Opposite signed seeds should attract. Same-role pairs are left untouched.
    comp=-(signed @ signed.T)
    sameb=(bid[0][:,None]==bid[0][None,:])
    nz=(sign[0]!=0)
    opposite=(sign[0][:,None]*sign[0][None,:] < 0)
    valid=sameb & nz[:,None] & nz[None,:] & opposite
    bias=torch.zeros_like(comp)
    bias[valid]=float(strength)*comp[valid].clamp_min(0)
    return bias

def chunk_logits(model,xc,p,k,c0,c1,q0,q1,attn_strength=0.0):
    h=model.emb(xc)+model.pos(p)
    mask=None if attn_strength==0 else cached_attn_bias(k,c0,c1,q0,q1,float(attn_strength))
    h=model.norm(model.tr(h,mask=mask))
    base=model.logits_from_repr(h)
    zs=[]
    for o in v.OFFSETS:
        if o==0: z=base
        else:
            ad=model.neighbor[str(o)]
            u=F.gelu(ad.a(h))
            vocab_lr=model.emb.weight @ ad.b.weight
            z=base + F.linear(u,vocab_lr)/math.sqrt(v.D)
        zs.append(z)
    return torch.stack(zs,dim=2)

def train_loss(model,x,y,m,attn_strength):
    losses=[];weights=[];B,L=x.shape
    for k,c0,c1,q0,q1 in v.ranges_for(L):
        xc=x[:,q0:q1];yc=y[:,q0:q1];mc=m[:,q0:q1]
        p,_,_=v.chunk_meta_paired(k,c0,c1,q0,q1,B)
        z=chunk_logits(model,xc,p,k,c0,c1,q0,q1,attn_strength)
        for hi,o in enumerate(v.OFFSETS):
            if o<0: pred=z[:,-o:,hi,:];tar=yc[:,:o];mm=mc[:,:o]
            elif o>0: pred=z[:,:-o,hi,:];tar=yc[:,o:];mm=mc[:,o:]
            else: pred=z[:,:,hi,:];tar=yc;mm=mc
            if mm.any():
                losses.append(F.cross_entropy(pred[mm],tar[mm]));weights.append(.60 if o==0 else .10)
    return sum(w*l for w,l in zip(weights,losses))/len(v.ranges_for(L))

def target_owner_seed(target_pos,boundary_id):
    # +S is the owner/core convention.
    return base_seed(target_pos,boundary_id)

@torch.no_grad()
def probs(model,x,attn_strength=0.0,link=True,link_gate=0.0):
    B,L=x.shape
    center=torch.zeros(B,L,v.VOCAB_SIZE)
    if not link:
        for k,c0,c1,q0,q1 in v.ranges_for(L):
            p,_,_=v.chunk_meta_paired(k,c0,c1,q0,q1,B)
            z=chunk_logits(model,x[:,q0:q1],p,k,c0,c1,q0,q1,attn_strength)
            pr=F.softmax(z,dim=-1)
            center[:,c0:c1]=pr[:,c0-q0:c1-q0,v.OFFSETS.index(0),:]
        return center
    agg=torch.zeros(B,L,v.VOCAB_SIZE);wsum=torch.zeros(B,L)
    for k,c0,c1,q0,q1 in v.ranges_for(L):
        p,bid,sign=v.chunk_meta_paired(k,c0,c1,q0,q1,B)
        z=chunk_logits(model,x[:,q0:q1],p,k,c0,c1,q0,q1,attn_strength)
        pr=F.softmax(z,dim=-1); conf=pr.max(-1).values; pos=torch.arange(q0,q1)
        signed=base_seed(p,bid)*sign.float().unsqueeze(-1)
        for hi,o in enumerate(v.OFFSETS):
            tgt=pos+o;valid=(tgt>=0)&(tgt<L);idx=tgt[valid].long()
            src=pr[:,valid,hi,:];cf=conf[:,valid,hi]; gate=torch.ones_like(cf)
            if link_gate!=0:
                ss=sign[:,valid];bb=bid[:,valid];sv=signed[:,valid]
                tp=tgt[valid].unsqueeze(0).expand(B,-1)
                owner=target_owner_seed(tp,bb)
                compat=-(sv*owner).sum(-1).clamp(-1,1)
                # Only borrowed halo proposals are explicitly routed; owner/internal stays neutral.
                gate=torch.where(ss<0,torch.exp(float(link_gate)*compat),gate)
            w=cf*gate
            agg.index_add_(1,idx,src*w.unsqueeze(-1));wsum.index_add_(1,idx,w)
    return agg/wsum.clamp_min(1e-9).unsqueeze(-1)

def batch(L,B,rep,base=31000):
    g=torch.Generator().manual_seed(v.SEED+base+L*31+rep)
    starts=torch.randint(0,len(v.val_ids)-L-1,(B,),generator=g)
    y=torch.stack([v.val_ids[int(s):int(s)+L] for s in starts])
    m=torch.rand(B,L,generator=g)<.60;x=y.clone();x[m]=v.MASK_ID
    return x,y,m

def stats_init(): return {'nll':0.,'bnll':0.,'inll':0.,'n':0,'bn':0,'inn':0,'ok':0,'bok':0}
def stats_add(S,p,y,m,L):
    py=p.gather(-1,y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9);lp=-py.log();pred=p.argmax(-1)
    bmask=torch.zeros(L,dtype=torch.bool)
    for b in range(v.SHARD,L,v.SHARD): bmask[max(0,b-v.HALO):min(L,b+v.HALO)]=True
    bm=m&bmask.unsqueeze(0);im=m&~bmask.unsqueeze(0)
    S['nll']+=float(lp[m].sum());S['bnll']+=float(lp[bm].sum());S['inll']+=float(lp[im].sum())
    S['n']+=int(m.sum());S['bn']+=int(bm.sum());S['inn']+=int(im.sum())
    S['ok']+=int((pred.eq(y)&m).sum());S['bok']+=int((pred.eq(y)&bm).sum())
def stats_fin(S):
    return {'nll':S['nll']/S['n'],'boundary_nll':S['bnll']/S['bn'],'interior_nll':S['inll']/S['inn'],'acc':S['ok']/S['n'],'boundary_acc':S['bok']/S['bn'],'n':S['n'],'boundary_n':S['bn']}

def evaluate(model,attn_strength,link_gate=0.0,reps=6):
    out={}
    model.eval()
    for L,B in [(256,16),(512,8),(1024,4)]:
        modes={'center':stats_init(),'linked':stats_init(),'router_off_linked':stats_init()}
        for r in range(reps):
            x,y,m=batch(L,B,r)
            stats_add(modes['center'],probs(model,x,attn_strength,False,0),y,m,L)
            stats_add(modes['linked'],probs(model,x,attn_strength,True,link_gate),y,m,L)
            stats_add(modes['router_off_linked'],probs(model,x,0.0,True,0.0),y,m,L)
        out[str(L)]={k:stats_fin(s) for k,s in modes.items()}
    return out

def train_run(name,attn_strength,steps):
    torch.manual_seed(v.SEED+12345);m=v.ParityDenoiser('clmp')
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=0.01)
    hist=[];seen={256:0,512:0,1024:0};t0=time.perf_counter()
    for step in range(1,steps+1):
        torch.manual_seed(v.SEED+777+step)
        L=v.CONTEXTS[(step-1)%3];B=v.BATCH_BY_L[L];seen[L]+=B*L
        x,mask,y=v.sample_sequences(v.train_ids,B,L)
        loss=train_loss(m,x,y,mask,attn_strength)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if step==1 or step%25==0 or step==steps:
            hist.append([step,L,float(loss.detach())]);print(name,'step',step,'L',L,'loss',float(loss.detach()),flush=True)
    sec=time.perf_counter()-t0
    ck={'state_dict':m.state_dict(),'attn_strength':attn_strength,'steps':steps,'params':v.count_params(m),'seen':seen,'hist':hist,'seconds':sec}
    torch.save(ck,os.path.join(BASE_DIR, f'{name}.pt'))
    return m,ck

if __name__=='__main__':
    name=sys.argv[1] if len(sys.argv)>1 else 'clmp_seed_router_v4'
    strength=float(sys.argv[2]) if len(sys.argv)>2 else 2.0
    steps=int(sys.argv[3]) if len(sys.argv)>3 else 300
    m,ck=train_run(name,strength,steps)
    # gate sweep is inference-only on same trained weights.
    gate_sweep={}
    for g in [0.0,0.25,0.5,1.0]:
        gate_sweep[str(g)]=evaluate(m,strength,g,reps=3)
        print('gate',g,gate_sweep[str(g)]['1024']['linked'],flush=True)
    report={'checkpoint':ck,'gate_sweep':gate_sweep}
    open(os.path.join(BASE_DIR, f'{name}_report.json'),'w').write(json.dumps(report,indent=2))
