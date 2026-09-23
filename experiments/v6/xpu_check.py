import torch, torch.nn.functional as F
from v6 import MDLM, ROOT
ck = torch.load(ROOT / 'v6_model.pt', map_location='cpu')
m = MDLM(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
torch.manual_seed(0); x = torch.randint(4, 1024, (2, 256)); x[:, ::3] = 1; p = torch.arange(256).expand(2, -1)
with torch.no_grad():
    zc = m(x, p)
    mx = m.to('xpu'); zx = mx(x.to('xpu'), p.to('xpu')).cpu()
    with torch.autocast('xpu', dtype=torch.bfloat16): zb = mx(x.to('xpu'), p.to('xpu')).float().cpu()
print('cpu vs xpu fp32 max|d|', float((zc - zx).abs().max()), '| cpu vs xpu bf16', float((zc - zb).abs().max()))
# does output depend on context at all? perturb other tokens, check position 100
x2 = x.clone(); x2[:, :90] = torch.randint(4, 1024, (2, 90))
with torch.no_grad(): zc2 = m.cpu()(x2, p)
print('logit change at pos 100 when tokens 0..89 change (cpu):', float((zc[:, 100] - zc2[:, 100]).abs().max()))
print('logit std across positions (cpu):', float(zc[0].std(0).mean()), '| emb norm', float(m.emb.weight.norm(dim=1).mean()), '| pos norm', float(m.pos.weight.norm(dim=1).mean()))
