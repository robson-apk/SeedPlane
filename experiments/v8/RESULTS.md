# SeedPlane V8 — resultado (2026-09-23)

**Veredito pré-registrado: SAME_OR_BETTER_QUALITY_AND_FASTER** (Q e S passaram). O critério foi publicado no git
(commit 2b3fbe5) antes da execução.

Mesmo checkpoint (`checkpoints/v6_mdlm_d256_l6_seed1.pt`), mesmas sequências, ordem sorteada, CPU do 5600X.
Traditional = 1 processo, atenção global, 4 threads. SeedPlane = 4 workers persistentes, shards de 128 + halo 16,
com IPC incluído. Decodificação iterativa de 16 passos, 50% mascarado, 32 sequências por seed.

## L = 1024
| Seed | Tempo trad. (mediana) | Tempo SeedPlane | Razão (IC95%) | Acurácia trad. | Acurácia SeedPlane | Δ (IC95%) |
|---|---|---|---|---|---|---|
| 11 | 1.549 ms | 762 ms | 0,491 [0,485; 0,498] | 44,19% | 44,69% | +0,50 pp [−0,02; +1,06] |
| 23 | 1.555 ms | 757 ms | 0,489 [0,481; 0,498] | 46,30% | 47,74% | +1,44 pp [+0,83; +2,07] |
| 37 | 1.559 ms | 762 ms | 0,487 [0,482; 0,493] | 44,21% | 44,58% | +0,37 pp [−0,08; +0,81] |

NLL de um passo (trad. → SeedPlane), média das seeds: 15% 1,647 → 1,604; 50% 2,495 → 2,455; 90% 4,260 → 4,266 (+0,15%).

## L = 512
Razão de tempo 0,90–0,94 (6–10% mais rápido); acurácia −0,57 a +0,23 pp (ICs cruzam 0); NLL igual ou melhor em
15%/50%, +0,4–0,5% em 90%.

## Escopo
TinyStories, onde o contexto relevante é local. Na V7, com dependência de longo alcance obrigatória, o SeedPlane
perde para a atenção global (acaso contra 100%). Ou seja: **2x mais rápido com qualidade igual ou melhor em texto
de dependência local**, sem substituir a atenção global quando é preciso ligar partes distantes.
