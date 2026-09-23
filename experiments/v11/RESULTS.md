# SeedPlane V11 — resultado (2026-09-23)

## Parte A — execução vetorizada na GPU (checkpoint V6, L=1.024): A1 FALHOU (por pouco) · A2 FALHOU · A3 PASSOU
| B580, 1.024 tokens | SeedPlane V10 | SeedPlane vetorizado | Tradicional |
|---|---|---|---|
| 1 página | 207–212 ms | **115–117 ms** | 101–103 ms |
| 32 páginas | 1.322 ms | 1.298 ms (12.700 tok/s) | 1.889 ms (8.700 tok/s) |
- A1 (≤ 0,5 × V10): razão 0,55 → falhou por pouco (1,8× mais rápido).
- A2 (mais rápido que o tradicional em 1 página): falhou; a distância caiu de 2,1× para 1,15×.
- A3 (concordância de tokens ≥ 99%): 99,5–99,8% → passou.

## Parte B — contexto longo: TREINO FALHOU no gate de sanidade
Modelo com posições até 8.192 (10.000 passos, curriculum de máscara da V6, gradient checkpointing, adendos 1–2):
loss com 15% de máscara = 5,03 (L=1.024) e 5,04 (L=8.192), unigrama 5,10, gate exigia ≤ 4,10 → **falhou**.
**B2 (qualidade) não é interpretável** e não é reportado como resultado.

## Descritivo válido mesmo com o treino falho: tempo × comprimento (1 página, B580, mesmas contas de qualquer peso)
| L | Tradicional | SeedPlane | Razão SeedPlane/Trad (IC95% superior, pior seed) |
|---|---|---|---|
| 1.024 | 104 ms (4.930 tok/s) | 119 ms (4.324 tok/s) | 1,140 (1,166) |
| 2.048 | 264 ms (3.862 tok/s) | 120 ms (8.430 tok/s) | 0,458 (0,490) |
| 4.096 | 960 ms (2.138 tok/s) | 155 ms (13.248 tok/s) | 0,161 (0,162) |
| 8.192 | 3.814 ms (1.072 tok/s) | 323 ms (12.665 tok/s) | 0,085 (0,085) |
O custo de execução não depende do valor dos pesos (mesmas multiplicações), por isso o tempo é informativo; o critério
B1 é registrado como "passaria", mas o protocolo manda reportar a falha de treino — a vitória de velocidade em texto
longo com qualidade comprovada fica para a V12 (Qwen2.5-0.5B pré-treinado).
Cruzamento na GPU: a partir de 2.048 tokens o SeedPlane é mais rápido; em 8.192, 11,8×.
