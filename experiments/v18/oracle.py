import sys, json, time, numpy as np, torch
sys.path.insert(0, r'C:\Users\Windows 11\sp_prof')
from qwen_engine import Qwen2Engine
B = r'C:\Users\Windows 11\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(6)
e = Qwen2Engine(B, 'cpu', torch.float32); ids = [9707, 11, 1879, 0]
c = e.new_cache(len(ids) + 130); lg, c = e.forward(ids, c)
np.asarray(lg[-1].numpy(), dtype=np.float32).tofile(r'C:\Users\Windows 11\sp_vk\oracle_logits.f32')
toks = []
for _ in range(128):
    t = int(lg[-1].argmax()); toks.append(t); lg, c = e.forward([t], c)
json.dump({'device': 'cpu', 'dtype': 'float32', 'tokens': toks}, open(r'C:\Users\Windows 11\sp_vk\oracle.json', 'w'))
print('oracle done', toks[:8])
