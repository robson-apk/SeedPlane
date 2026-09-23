# SeedPlane V11 — SeedPlane na GPU: execução vetorizada (A) + contexto longo até 8.192 tokens (B)
Protocolo antes dos resultados, 2026-09-23. Hardware: Intel Arc B580 (XPU) no Ryzen 5600X.

## Motivação (V10)
Na B580, uma página de 1.024 tokens: tradicional 93 ms × SeedPlane 199 ms. O SeedPlane da V10 roda na GPU em vários
lotes pequenos (agrupados por comprimento de janela) com laços em Python por shard e sincronização CPU↔GPU a cada passo.

## Parte A — execução vetorizada (checkpoint V6, L=1024)
- `sp_gpu`: todas as janelas de todas as páginas num único lote de largura 160 (bordas completadas com padding +
  `src_key_padding_mask`), núcleos extraídos por fatia fixa, softmax/argmax e a regra de commit (cota por shard) na GPU;
  sincronização só no fim.
- `trad_gpu`: atenção global, softmax/argmax e commit na GPU, sincronização só no fim (mesmo tratamento).
- `sp_v10`: a implementação da V10 (lotes por comprimento, laço por shard) como referência.
Modos: 1 página (latência) e 32 páginas (vazão). 3 seeds × 8 páginas (latência) / 3 lotes (vazão). 16 passos, 50% mascarado.
Critérios:
- **A1:** latência `sp_gpu` ≤ 0,5 × `sp_v10` (3 seeds, IC superior da razão < 0,5).
- **A2:** latência `sp_gpu` < `trad_gpu` (IC superior < 1), 3 seeds. Previsão registrada: incerta.
- **A3:** concordância de tokens `sp_gpu` × `sp_v10` ≥ 99% (padding e desempates podem mudar casos raros).

## Parte B — contexto longo
Modelo novo com a mesma arquitetura da V6 (d=256, 6 camadas) e tabela de posições até 8.192; treino na B580 com curriculum
de máscara (V6) e comprimentos misturados {1.024, 2.048, 4.096, 8.192} + janelas de shard em posições absolutas até 8.192.
Um treino (seed 1); gate de sanidade antes de qualquer critério: loss com 15% de máscara em L=8.192 ≤ unigrama − 1,0.
Se o gate falhar: reportar falha de treino.
Head-to-head na GPU (`sp_gpu` × `trad_gpu`, 1 página) em L ∈ {1.024, 2.048, 4.096, 8.192}; 3 seeds × 6 páginas.
Critérios:
- **B1 (velocidade em texto longo):** em L=8.192, latência `sp_gpu` ≤ 0,90 × `trad_gpu` (IC superior < 0,90), 3 seeds.
- **B2 (qualidade em texto longo):** em L=8.192, acurácia `sp_gpu` − `trad_gpu` com limite inferior do IC > −0,5 pp, 3 seeds.
- Descritivo: comprimento de cruzamento (menor L em que `sp_gpu` fica mais rápido) e curva de tempo × L.
Escopo: TinyStories concatenado (dependência longa fraca — a V7 mostrou que o SeedPlane não recupera informação
distante); modelo de 5,3M; uma GPU.
