# SeedPlane V9 — otimizações estruturais + escala de 1 a 6 núcleos (protocolo antes dos resultados, 2026-09-23)

## Perfil que motivou (exploratório, `profile_v8.py`, `results/profile_v8.json`, L=1024, Ryzen 5600X)
Forward global: 210 ms (1 thread) → 90 ms (4 threads) = só 2,3× com 4 núcleos. Um shard (160 tokens): 18 ms.
Passo da V8 com 4 processos de 1 thread: 47,6 ms (≈ 4× de eficiência). IPC: 1,2 ms/passo.
Leitura: o ganho da V8 vem sobretudo de paralelismo embaraçoso (processos independentes) contra o paralelismo
intra-operação do PyTorch, não da economia de atenção (o halo acrescenta 25% de tokens).

## Otimizações (aplicadas ao SeedPlane; a que também se aplica ao tradicional é aplicada nos dois)
- O1 (ambos): logits só nas posições mascaradas (projeção de vocabulário + softmax só onde se decide).
- O2 (SeedPlane): shards de um worker num único forward em lote (bordas com padding + key_padding_mask).
- O3 (SeedPlane): distribuição dos shards por custo (LPT: shards de borda custam 144, internos 160).

## Condições (mesmo checkpoint V6, CPU do 5600X, nada mais rodando)
Para c ∈ {1, 2, 3, 4, 5, 6} núcleos:
- `trad`: 1 processo, `torch.set_num_threads(c)`, com O1.
- `sp_v8`: implementação da V8 (c workers × 1 thread), sem mudanças.
- `sp_opt`: c workers × 1 thread, com O1+O2+O3.
L ∈ {512, 1024}; 16 passos; 50% mascarado; 3 seeds × 12 sequências; ordem sorteada.

## Critérios (fixados agora)
- **P1 (vantagem em todo c):** em L=1024, `sp_opt` mais rápido que `trad` em cada c de 1 a 6 (limite superior do
  IC95% da razão de tempo < 1), nas 3 seeds.
- **P2 (ganho das otimizações):** em L=1024, c=4: tempo `sp_opt` ≤ 0,90 × `sp_v8` (IC superior < 0,90), 3 seeds.
- **P3 (qualidade preservada):** `sp_opt` × `sp_v8`: concordância dos tokens finais ≥ 99% e diferença de acurácia
  dentro de ±0,3 pp, nas 3 seeds.
Descritivo (sem critério): curvas de aceleração T(1)/T(c) e eficiência paralela de cada método.
Previsão registrada: com 8 shards, c=5 e c=6 ganham pouco sobre c=4 no SeedPlane (desbalanceamento: alguém fica com 2
shards). Escopo: modelo de 5,3M, L ≤ 1024 (limite de posições do checkpoint), CPU.

## Adendo 1 — O2 trocada antes da rodada de critérios
Smoke test (1 sequência, c ∈ {1, 2}, L=512): `sp_opt` com lote + padding + key_padding_mask ficou MAIS LENTO que `sp_v8`
(1.273 contra 1.185 ms em c=1), com concordância de tokens de 100%. Microbenchmark (`micro_batching.py`, 4 shards, 1 thread):
separados 70,5 ms; só posições mascaradas 65,9; lote com padding+máscara 75,3; **lote por comprimento igual, sem padding,
61,5**. A O2 passa a ser "lote por comprimento igual". Critérios inalterados; nenhum dado de critério tinha sido coletado.
