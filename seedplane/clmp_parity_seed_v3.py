import os, re, math, json, random, time, collections
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=20260922
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(5)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
_candidate_corpus = [
    os.path.join(REPO_DIR, '..', '..', 'CORPUS', 'TinyStories-valid.txt'),
    os.path.join(REPO_DIR, 'data', 'TinyStories-valid.txt'),
    '/mnt/data/TinyStories-valid(5).txt'
]
CORPUS = next((p for p in _candidate_corpus if os.path.exists(p)), _candidate_corpus[0])
VOCAB_SIZE=1024; MASK_ID=1; UNK_ID=2; EOS_ID=3
MAX_SEQ=1024; CONTEXTS=(256,512,1024)
D=32; HEADS=4; LAYERS=2; FF=64
SHARD=128; HALO=16
OFFSETS=(-2,-1,0,1,2)
LOW_RANK=4
MASK_RATE=(0.25,0.85)
SEED_SCALE=0.35
DEVICE='cpu'

# corpus/tokenizer (cached after first parse)
CACHE=os.path.join(REPO_DIR, 'data', 'tinystories_word1024_cache.pt')
pat=re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]",re.I)
def tok(s): return [t.lower() for t in pat.findall(s)]
if os.path.exists(CACHE):
    c=torch.load(CACHE,map_location='cpu')
    vocab=c['vocab']; stoi={w:i for i,w in enumerate(vocab)}
    train_ids=c['train_ids']; val_ids=c['val_ids']; coverage=c['coverage']; stories_count=c['stories_count']
    stories=[None]*stories_count
else:
    raw=open(CORPUS,'r',encoding='utf-8',errors='ignore').read()
    stories_raw=[x.strip() for x in raw.split('<|endoftext|>') if x.strip()]
    cut=int(len(stories_raw)*0.90)
    train_stories,val_stories=stories_raw[:cut],stories_raw[cut:]
    cnt=collections.Counter()
    for x in train_stories: cnt.update(tok(x))
    vocab=['<pad>','<mask>','<unk>','<eos>']+[w for w,_ in cnt.most_common(VOCAB_SIZE-4)]
    stoi={w:i for i,w in enumerate(vocab)}
    def encode(ss):
        out=[]
        for x in ss:
            out.extend(stoi.get(t,UNK_ID) for t in tok(x)); out.append(EOS_ID)
        return torch.tensor(out,dtype=torch.long)
    train_ids=encode(train_stories); val_ids=encode(val_stories)
    coverage=sum(cnt[w] for w in vocab[4:])/sum(cnt.values())
    stories_count=len(stories_raw); stories=[None]*stories_count
    torch.save({'vocab':vocab,'train_ids':train_ids,'val_ids':val_ids,'coverage':coverage,'stories_count':stories_count},CACHE)

class LowRankResidual(nn.Module):
    def __init__(self,d,rank):
        super().__init__()
        self.a=nn.Linear(d,rank,bias=False)
        self.b=nn.Linear(rank,d,bias=False)
        nn.init.normal_(self.a.weight,std=0.02)
        nn.init.zeros_(self.b.weight)
    def forward(self,h): return self.b(F.gelu(self.a(h)))

class ParityDenoiser(nn.Module):
    """Exact head-param parity.
    global: one rank=4*r adapter for center -> 2*d*(4r)=8dr params
    clmp: four rank=r neighbor adapters -> 4*(2dr)=8dr params; center uses h directly.
    All logits tied to token embedding matrix.
    """
    def __init__(self,kind='global'):
        super().__init__(); assert kind in ('global','clmp'); self.kind=kind
        self.emb=nn.Embedding(VOCAB_SIZE,D)
        self.pos=nn.Embedding(MAX_SEQ,D)
        layer=nn.TransformerEncoderLayer(D,HEADS,FF,dropout=0.0,batch_first=True,norm_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(layer,LAYERS)
        self.norm=nn.LayerNorm(D)
        self.out_bias=nn.Parameter(torch.zeros(VOCAB_SIZE))
        if kind=='global':
            self.center_adapter=LowRankResidual(D,4*LOW_RANK)
            self.neighbor=nn.ModuleDict()
        else:
            self.center_adapter=None
            self.neighbor=nn.ModuleDict({str(o):LowRankResidual(D,LOW_RANK) for o in OFFSETS if o!=0})
    def procedural_seed(self,pos_ids,boundary_ids,halo_sign,mode):
        # Parameter-free deterministic field. Same boundary shares phase; complement mode flips sign across the boundary.
        # pos_ids/boundary_ids/halo_sign: [B,L], halo_sign in {-1,0,+1}
        B,L=pos_ids.shape
        half=D//2
        j=torch.arange(half,device=pos_ids.device,dtype=torch.float32).view(1,1,-1)
        p=pos_ids.float().unsqueeze(-1)
        b=boundary_ids.float().unsqueeze(-1)
        freq=torch.exp(-math.log(10000.0)*j/max(1,half-1))
        phase=(p+17.0*b+0.5)*freq
        s=torch.cat([torch.sin(phase),torch.cos(phase)],dim=-1)
        if s.size(-1)<D: s=F.pad(s,(0,D-s.size(-1)))
        s=s[...,:D]
        mask=(halo_sign!=0).float().unsqueeze(-1)
        if mode=='shared': mult=mask
        elif mode in ('complement','paired'): mult=halo_sign.float().unsqueeze(-1)
        else: return torch.zeros(B,L,D,device=pos_ids.device)
        return SEED_SCALE*s*mult
    def hidden(self,x,pos_ids,seed_mode='none',boundary_ids=None,halo_sign=None):
        h=self.emb(x)+self.pos(pos_ids)
        if seed_mode!='none':
            h=h+self.procedural_seed(pos_ids,boundary_ids,halo_sign,seed_mode)
        return self.norm(self.tr(h))
    def logits_from_repr(self,r):
        return F.linear(r,self.emb.weight,self.out_bias)/math.sqrt(D)
    def forward(self,x,pos_ids,seed_mode='none',boundary_ids=None,halo_sign=None):
        h=self.hidden(x,pos_ids,seed_mode,boundary_ids,halo_sign)
        if self.kind=='global':
            return self.logits_from_repr(h+self.center_adapter(h))
        # Exact low-rank factorization: compute the shared full-vocab base once.
        # For neighbor o: logits(h + B gelu(Ah)) = base + gelu(Ah) @ (E B)^T.
        # This preserves logits exactly while replacing four D×V projections by r×V deltas.
        base=self.logits_from_repr(h)
        zs=[]
        for o in OFFSETS:
            if o==0:
                z=base
            else:
                ad=self.neighbor[str(o)]
                u=F.gelu(ad.a(h))                    # [B,L,r]
                vocab_lr=self.emb.weight @ ad.b.weight # [V,r]
                delta=F.linear(u,vocab_lr)/math.sqrt(D)
                z=base+delta
            zs.append(z)
        return torch.stack(zs,dim=2) # B,L,H,V

def count_params(m): return sum(p.numel() for p in m.parameters())

def corrupt(y):
    B,L=y.shape
    rates=MASK_RATE[0]+(MASK_RATE[1]-MASK_RATE[0])*torch.rand(B,1)
    m=torch.rand(B,L)<rates
    x=y.clone(); x[m]=MASK_ID
    return x,m

def sample_sequences(ids,B,L):
    starts=torch.randint(0,len(ids)-L-1,(B,))
    y=torch.stack([ids[int(s):int(s)+L] for s in starts])
    return (*corrupt(y), y) # x,m,y

def ranges_for(L):
    out=[]; k=0
    for c0 in range(0,L,SHARD):
        c1=min(L,c0+SHARD); q0=max(0,c0-HALO); q1=min(L,c1+HALO)
        out.append((k,c0,c1,q0,q1)); k+=1
    return out

def chunk_meta(k,c0,c1,q0,q1,B):
    pos=torch.arange(q0,q1).unsqueeze(0).expand(B,-1)
    sign=torch.zeros(B,q1-q0,dtype=torch.long)
    bid=torch.zeros(B,q1-q0,dtype=torch.long)
    # left halo belongs to boundary c0; right halo boundary c1. Signs are complementary between adjacent chunks.
    if q0<c0:
        n=c0-q0; sign[:,:n]=-1; bid[:,:n]=c0//SHARD
    if c1<q1:
        n=q1-c1; sign[:,-n:]=+1; bid[:,-n:]=c1//SHARD
    return pos,bid,sign


def chunk_meta_paired(k,c0,c1,q0,q1,B):
    """Exact complementary overlap: owned boundary edge +S, borrowed halo -S.
    The same global token seen by adjacent chunks receives opposite seed vectors.
    """
    pos=torch.arange(q0,q1).unsqueeze(0).expand(B,-1)
    sign=torch.zeros(B,q1-q0,dtype=torch.long)
    bid=torch.zeros(B,q1-q0,dtype=torch.long)
    # left boundary at c0 (if not first shard)
    if q0<c0:
        b=c0//SHARD
        n=c0-q0; sign[:,:n]=-1; bid[:,:n]=b
        e=min(HALO,c1-c0); a=c0-q0; sign[:,a:a+e]=+1; bid[:,a:a+e]=b
    # right boundary at c1 (if not last shard)
    if c1<q1:
        b=c1//SHARD
        e=min(HALO,c1-c0); a=c1-e-q0; sign[:,a:a+e]=+1; bid[:,a:a+e]=b
        n=q1-c1; sign[:,-n:]=-1; bid[:,-n:]=b
    return pos,bid,sign
def global_loss(model,x,y,m):
    B,L=x.shape; p=torch.arange(L).unsqueeze(0).expand(B,-1)
    z=model(x,p)
    return F.cross_entropy(z[m],y[m])

def clmp_train_loss(model,x,y,m,seed_mode='none'):
    # Train on every shard+halo; center gets 50% total objective, 4 neighbors split remaining 50%.
    losses=[]; weights=[]
    B,L=x.shape
    for k,c0,c1,q0,q1 in ranges_for(L):
        xc=x[:,q0:q1]; yc=y[:,q0:q1]; mc=m[:,q0:q1]
        p,bid,sign=(chunk_meta_paired(k,c0,c1,q0,q1,B) if seed_mode=='paired' else chunk_meta(k,c0,c1,q0,q1,B))
        z=model(xc,p,seed_mode,bid,sign)
        for hi,o in enumerate(OFFSETS):
            if o<0:
                pred=z[:,-o:,hi,:]; tar=yc[:,:o]; mm=mc[:,:o]
            elif o>0:
                pred=z[:,:-o,hi,:]; tar=yc[:,o:]; mm=mc[:,o:]
            else:
                pred=z[:,:,hi,:]; tar=yc; mm=mc
            if mm.any():
                losses.append(F.cross_entropy(pred[mm],tar[mm])); weights.append(0.60 if o==0 else 0.10)
    # normalize across chunks so objective scale doesn't grow with #shards
    nch=len(ranges_for(L))
    return sum(w*l for w,l in zip(weights,losses))/nch

BATCH_BY_L={256:8,512:4,1024:2}
def train(model,kind,seed_mode='none',steps=240,log_every=40):
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=0.01)
    hist=[]; seen={256:0,512:0,1024:0}; t0=time.perf_counter()
    schedule=[256,512,1024]
    for step in range(1,steps+1):
        # equal number of updates at each real context length
        L=schedule[(step-1)%3]; B=BATCH_BY_L[L]; seen[L]+=B*L
        x,m,y=sample_sequences(train_ids,B,L)
        loss=global_loss(model,x,y,m) if kind=='global' else clmp_train_loss(model,x,y,m,seed_mode)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%log_every==0 or step==steps:
            hist.append([step,L,float(loss.detach())]); print(kind,seed_mode,'step',step,'L',L,'loss',float(loss.detach()),flush=True)
    return {'seconds':time.perf_counter()-t0,'loss_curve':hist,'tokens_seen_by_context':seen,'total_tokens_seen':sum(seen.values())}

@torch.no_grad()
def global_probs(model,x):
    B,L=x.shape;p=torch.arange(L).unsqueeze(0).expand(B,-1)
    return F.softmax(model(x,p),dim=-1)

@torch.no_grad()
def clmp_probs(model,x,seed_mode='none',link=True):
    B,L=x.shape
    agg=torch.zeros(B,L,VOCAB_SIZE); wsum=torch.zeros(B,L)
    votes=torch.zeros(B,L,VOCAB_SIZE); proposal_count=torch.zeros(L)
    center=torch.zeros(B,L,VOCAB_SIZE)
    for k,c0,c1,q0,q1 in ranges_for(L):
        xc=x[:,q0:q1]; p,bid,sign=(chunk_meta_paired(k,c0,c1,q0,q1,B) if seed_mode=='paired' else chunk_meta(k,c0,c1,q0,q1,B))
        z=model(xc,p,seed_mode,bid,sign); probs=F.softmax(z,dim=-1)
        # center-only route for owned core region
        center[:,c0:c1]=probs[:,c0-q0:c1-q0,OFFSETS.index(0),:]
        if not link: continue
        conf,top=probs.max(-1)
        pos=torch.arange(q0,q1)
        for hi,o in enumerate(OFFSETS):
            tgt=pos+o; valid=(tgt>=0)&(tgt<L)
            idx=tgt[valid].long(); src=probs[:,valid,hi,:]; cf=conf[:,valid,hi]
            agg.index_add_(1,idx,src*cf.unsqueeze(-1)); wsum.index_add_(1,idx,cf)
            one=F.one_hot(top[:,valid,hi],VOCAB_SIZE).float()*cf.unsqueeze(-1)
            votes.index_add_(1,idx,one); proposal_count.index_add_(0,idx,torch.ones_like(idx,dtype=torch.float))
    if not link: return center,None,None,proposal_count
    final=agg/wsum.clamp_min(1e-9).unsqueeze(-1)
    winner=final.argmax(-1)
    agree=votes.gather(-1,winner.unsqueeze(-1)).squeeze(-1)/wsum.clamp_min(1e-9)
    conf=final.max(-1).values*agree
    return final,conf,agree,proposal_count

def fixed_val(L,B=8,mask_rate=.60):
    g=torch.Generator().manual_seed(SEED+L)
    starts=torch.randint(0,len(val_ids)-L-1,(B,),generator=g)
    y=torch.stack([val_ids[int(s):int(s)+L] for s in starts])
    m=torch.rand(B,L,generator=g)<mask_rate; x=y.clone(); x[m]=MASK_ID
    return x,y,m

def score(probs,y,m,L):
    pred=probs.argmax(-1); ok=(pred==y)
    # boundary +/- 4 around each shard cut
    bmask=torch.zeros(L,dtype=torch.bool)
    for b in range(SHARD,L,SHARD): bmask[max(0,b-4):min(L,b+4)]=True
    bm=m & bmask.unsqueeze(0); im=m & (~bmask.unsqueeze(0))
    py=probs.gather(-1,y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
    return {
      'masked_acc':float((ok&m).sum()/m.sum()),
      'boundary_acc':float((ok&bm).sum()/bm.sum()) if bm.any() else None,
      'interior_acc':float((ok&im).sum()/im.sum()),
      'nll':float((-py[m].log()).mean()),
      'boundary_n':int(bm.sum()), 'masked_n':int(m.sum())
    }

@torch.no_grad()
def benchmark(model,kind,seed_mode,L,reps=8):
    B=1; x=torch.full((B,L),MASK_ID,dtype=torch.long)
    old=torch.get_num_threads(); torch.set_num_threads(1)
    if kind=='global':
        f=lambda: global_probs(model,x)
    else:
        f=lambda: clmp_probs(model,x,seed_mode,True)[0]
    for _ in range(2): f()
    t=time.perf_counter()
    for _ in range(reps): f()
    ms=(time.perf_counter()-t)*1000/reps
    torch.set_num_threads(old)
    return ms

def main(steps=240):
    models={
      'global':ParityDenoiser('global'),
      'clmp':ParityDenoiser('clmp'),
      'clmp_seed_shared':ParityDenoiser('clmp'),
      'clmp_seed_complement':ParityDenoiser('clmp'),
    }
    params={k:count_params(v) for k,v in models.items()}
    print('PARAMS',params,flush=True)
    stats={}
    stats['global']=train(models['global'],'global','none',steps)
    stats['clmp']=train(models['clmp'],'clmp','none',steps)
    stats['clmp_seed_shared']=train(models['clmp_seed_shared'],'clmp','shared',steps)
    stats['clmp_seed_complement']=train(models['clmp_seed_complement'],'clmp','complement',steps)
    metrics={}
    meta={}
    for L in CONTEXTS:
        B={256:24,512:12,1024:6}[L]
        x,y,m=fixed_val(L,B)
        metrics[str(L)]={}
        pg=global_probs(models['global'],x); metrics[str(L)]['global']=score(pg,y,m,L)
        for name,sm in [('clmp','none'),('clmp_seed_shared','shared'),('clmp_seed_complement','complement')]:
            p0,_,_,pc0=clmp_probs(models[name],x,sm,False)
            pl,cf,ag,pc=clmp_probs(models[name],x,sm,True)
            metrics[str(L)][name+'_center']=score(p0,y,m,L)
            metrics[str(L)][name+'_linked']=score(pl,y,m,L)
            meta.setdefault(name,{})[str(L)]={
              'mean_agreement':float(ag[m].mean()),'mean_linked_conf':float(cf[m].mean()),
              'mean_proposals':float(pc.mean()),'boundary_mean_proposals':float(torch.stack([pc[max(0,b-4):min(L,b+4)].mean() for b in range(SHARD,L,SHARD)]).mean()) if L>SHARD else None
            }
    latency={}
    for L in CONTEXTS:
        latency[str(L)]={
          'global_1thread_ms':benchmark(models['global'],'global','none',L),
          'clmp_linked_1thread_ms':benchmark(models['clmp'],'clmp','none',L),
          'seed_complement_linked_1thread_ms':benchmark(models['clmp_seed_complement'],'clmp','complement',L),
        }
    # Verify exact complement construction numerically for one shared boundary.
    k0,c00,c01,q00,q01=ranges_for(256)[0]; k1,c10,c11,q10,q11=ranges_for(256)[1]
    p0,b0,s0=chunk_meta(k0,c00,c01,q00,q01,1); p1,b1,s1=chunk_meta(k1,c10,c11,q10,q11,1)
    seed0=models['clmp_seed_complement'].procedural_seed(p0,b0,s0,'complement')[0]
    seed1=models['clmp_seed_complement'].procedural_seed(p1,b1,s1,'complement')[0]
    # same positions in overlap [112,143], compare right halo of chunk0 vs left halo chunk1 over boundary-adjacent common positions 112:128 / 128:144 aren't same positions. Instead construction sign property is side-specific; document mean norms.
    seed_check={'nonzero_norm_chunk0':float(seed0.norm(dim=-1).mean()),'nonzero_norm_chunk1':float(seed1.norm(dim=-1).mean())}
    report={
      'config':{'contexts':CONTEXTS,'max_seq':MAX_SEQ,'shard':SHARD,'halo':HALO,'offsets':OFFSETS,'vocab':VOCAB_SIZE,'d_model':D,'layers':LAYERS,'heads':HEADS,'ff':FF,'low_rank':LOW_RANK,'steps_each':steps,'seed_scale':SEED_SCALE,'seed':SEED},
      'corpus':{'stories':len(stories),'train_tokens':len(train_ids),'val_tokens':len(val_ids),'vocab_coverage':coverage},
      'params':params,'train':stats,'metrics':metrics,'meta':meta,'latency':latency,'seed_check':seed_check
    }
    for name,m in models.items():
        torch.save({'state_dict':m.state_dict(),'vocab':vocab,'config':report['config'],'kind':'global' if name=='global' else 'clmp','seed_mode': {'global':'none','clmp':'none','clmp_seed_shared':'shared','clmp_seed_complement':'complement'}[name]},f'/mnt/data/{name}_parity_ctx1024.pt')
    open('/mnt/data/clmp_parity_seed_report.json','w').write(json.dumps(report,indent=2))
    print('FINAL_REPORT',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    steps=int(os.environ.get('STEPS','240')); main(steps)
