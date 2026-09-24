"""SeedPlane's native Qwen2 decoder.

This module loads safetensors directly and implements the Qwen2 graph, KV cache,
sampling and streaming with PyTorch tensor primitives.  It deliberately does
not instantiate a Transformers model or call the llama.cpp runtime.  PyTorch is
currently the portable reference backend; optimized SeedPlane backends must
match this implementation token-for-token before they are enabled.
"""
from dataclasses import dataclass
import json, math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class KVCache:
    keys: list
    values: list
    length: int = 0
    capacity: int = 0

    @classmethod
    def allocate(cls, layers, capacity, kv_heads, head_dim, device, dtype):
        shape = (1, capacity, kv_heads, head_dim)
        return cls([torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)],
                   [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)], 0, capacity)


class Qwen2Engine:
    """Minimal inference-only Qwen2/Qwen2.5 runtime with a persistent KV cache."""

    def __init__(self, bundle, device='cpu', dtype=None):
        self.bundle = Path(bundle)
        self.config = json.loads((self.bundle / 'config.json').read_text())
        if self.config.get('model_type') != 'qwen2':
            raise ValueError('SeedPlane Qwen2Engine only supports model_type=qwen2')
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type != 'cpu' else torch.float32
        self.dtype = dtype
        self.n_layers = self.config['num_hidden_layers']
        self.n_heads = self.config['num_attention_heads']
        self.n_kv_heads = self.config['num_key_value_heads']
        self.hidden = self.config['hidden_size']
        self.head_dim = self.hidden // self.n_heads
        if self.hidden % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError('invalid Qwen2 head configuration')
        theta = self.config.get('rope_theta', self.config.get('rope_parameters', {}).get('rope_theta', 10000.0))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        self.eps = self.config.get('rms_norm_eps', 1e-6)
        files = sorted(self.bundle.glob('*.safetensors'))
        if not files: raise FileNotFoundError(f'no safetensors in {self.bundle}')
        self.w = {}
        for filename in files:
            with safe_open(filename, framework='pt', device='cpu') as f:
                for name in f.keys():
                    if name in self.w: raise ValueError(f'duplicate tensor {name}')
                    self.w[name] = f.get_tensor(name).to(device=self.device, dtype=self.dtype)
        required = {'model.embed_tokens.weight', 'model.norm.weight'}
        if not required.issubset(self.w): raise ValueError('bundle is missing Qwen2 tensors')
        # SeedPlane-owned inference packing: reduce per-token GPU launches by
        # combining projections that consume the same activation.
        self.fused = {}; self.norm_w = {'model.norm.weight': self.w['model.norm.weight'].float()}
        for layer in range(self.n_layers):
            p = f'model.layers.{layer}'
            self.norm_w[p + '.input_layernorm.weight'] = self.w[p + '.input_layernorm.weight'].float()
            self.norm_w[p + '.post_attention_layernorm.weight'] = self.w[p + '.post_attention_layernorm.weight'].float()
            self.fused[p + '.self_attn.qkv_proj.weight'] = torch.cat([
                self.w[p + '.self_attn.q_proj.weight'], self.w[p + '.self_attn.k_proj.weight'], self.w[p + '.self_attn.v_proj.weight']])
            self.fused[p + '.self_attn.qkv_proj.bias'] = torch.cat([
                self.w[p + '.self_attn.q_proj.bias'], self.w[p + '.self_attn.k_proj.bias'], self.w[p + '.self_attn.v_proj.bias']])
            self.fused[p + '.mlp.gate_up_proj.weight'] = torch.cat([
                self.w[p + '.mlp.gate_proj.weight'], self.w[p + '.mlp.up_proj.weight']])
            for suffix in ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                           'self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias',
                           'mlp.gate_proj.weight', 'mlp.up_proj.weight'):
                del self.w[p + '.' + suffix]

    def new_cache(self, capacity=4096):
        capacity = min(capacity, self.config['max_position_embeddings'])
        if capacity < 1: raise ValueError('cache capacity must be positive')
        return KVCache.allocate(self.n_layers, capacity, self.n_kv_heads, self.head_dim, self.device, self.dtype)

    def _linear(self, x, name):
        return F.linear(x, self.w[name + '.weight'], self.w.get(name + '.bias'))

    def _norm(self, x, name):
        xf = x.float(); z = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (z * self.norm_w[name]).to(self.dtype)

    @staticmethod
    def _rotate_half(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    def _rope(self, q, k, positions):
        phase = torch.outer(positions.float(), self.inv_freq)
        phase = torch.cat((phase, phase), dim=-1)[None, :, None, :]
        cos, sin = phase.cos().to(self.dtype), phase.sin().to(self.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, all_logits=False):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0: raise ValueError('input_ids cannot be empty')
        if cache is None: cache = self.new_cache()
        start, qlen = cache.length, ids.numel()
        if start + qlen > cache.capacity: raise ValueError(f'KV cache capacity {cache.capacity} exceeded')
        positions = torch.arange(start, start + qlen, device=self.device)
        x = F.embedding(ids, self.w['model.embed_tokens.weight'])[None]
        for layer in range(self.n_layers):
            p = f'model.layers.{layer}'
            residual = x; h = self._norm(x, p + '.input_layernorm.weight')
            qkv = F.linear(h, self.fused[p + '.self_attn.qkv_proj.weight'], self.fused[p + '.self_attn.qkv_proj.bias'])
            q, k, v = qkv.split((self.hidden, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim), dim=-1)
            q = q.view(1, qlen, self.n_heads, self.head_dim)
            k = k.view(1, qlen, self.n_kv_heads, self.head_dim); v = v.view(1, qlen, self.n_kv_heads, self.head_dim)
            q, k = self._rope(q, k, positions)
            cache.keys[layer][:, start:start + qlen].copy_(k); cache.values[layer][:, start:start + qlen].copy_(v)
            k = cache.keys[layer][:, :start + qlen]; v = cache.values[layer][:, :start + qlen]
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            qt, kt, vt = (z.transpose(1, 2) for z in (q, k, v))
            if start == 0:
                attn = F.scaled_dot_product_attention(qt, kt, vt, is_causal=qlen > 1)
            else:
                allowed = torch.arange(start + qlen, device=self.device)[None, :] <= positions[:, None]
                attn = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=allowed)
            x = residual + self._linear(attn.transpose(1, 2).reshape(1, qlen, self.hidden), p + '.self_attn.o_proj')
            residual = x; h = self._norm(x, p + '.post_attention_layernorm.weight')
            gate, up = F.linear(h, self.fused[p + '.mlp.gate_up_proj.weight']).chunk(2, dim=-1)
            x = residual + self._linear(F.silu(gate) * up, p + '.mlp.down_proj')
        cache.length += qlen
        x = self._norm(x, 'model.norm.weight')
        head = self.w.get('lm_head.weight', self.w['model.embed_tokens.weight'])
        logits = F.linear(x if all_logits else x[:, -1:], head).float()[0]
        return logits, cache

    @staticmethod
    def sample(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None):
        if temperature <= 0: return int(logits.argmax())
        z = logits / temperature
        if top_k > 0:
            cutoff = torch.topk(z, min(top_k, z.numel())).values[-1]; z = z.masked_fill(z < cutoff, -torch.inf)
        if top_p < 1:
            values, indices = torch.sort(z, descending=True); probs = values.softmax(-1)
            drop = probs.cumsum(-1) - probs > top_p; values = values.masked_fill(drop, -torch.inf)
            z = torch.full_like(z, -torch.inf).scatter(0, indices, values)
        return int(torch.multinomial(z.softmax(-1), 1, generator=generator))

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0, eos_token_id=None, seed=1):
        cache = self.new_cache(len(prompt_ids) + max_new_tokens); logits, cache = self.forward(prompt_ids, cache)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        for _ in range(max_new_tokens):
            token = self.sample(logits[-1], temperature, top_k, top_p, gen)
            yield token
            if token == eos_token_id: break
            logits, cache = self.forward([token], cache)


def chat_prompt(messages):
    """Render Qwen's ChatML without depending on Transformers."""
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + '<|im_start|>assistant\n'
