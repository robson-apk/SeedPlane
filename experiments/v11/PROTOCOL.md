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

## Adendo 1 — parte A concluída; ajuste de memória no treino da parte B (antes de treinar)
Sonda de memória na B580: um passo de treino com L=8.192 (batch 1) estoura a VRAM com atenção global materializada
(2 GiB por alocação); com gradient checkpointing por camada cabe (6,2 GB, 0,66 s/passo). O treino usa checkpointing
para L ≥ 4.096. É só uma técnica de memória (recalcula ativações no backward; a conta é a mesma). A VRAM livre da B580
agora é ~11 GB (antes ~3 GB). Critérios da parte B inalterados.

## Adendo 2 — treino relançado (antes de qualquer resultado da parte B)
O primeiro treino rodou a ~2 s/passo (5,5 h previstas) e foi interrompido no passo ~300. Perfil por tipo de passo:
L=1.024 0,10 s · 2.048 0,13 s · 4.096 0,33 s · **8.192 15,3 s** · janelas 0,04–0,05 s. Com o cache do alocador ocupado
por outros formatos, a atenção global de 8.192² transbordava para a memória do sistema. Liberar o cache
(`torch.xpu.empty_cache()`) antes dos passos de 8.192 → 0,77 s. Só isso mudou; receita e critérios inalterados.
