# SeedPlane V9 — resultado (2026-09-23)

**P1 PASSOU · P2 FALHOU · P3 PASSOU.** Critérios publicados no git (commit 7ff88e1) antes da execução.

## L = 1024 (mediana de 36 sequências por célula; tokens/s = tokens escondidos preenchidos por segundo)
| Núcleos | Transformer tradicional | SeedPlane V8 | SeedPlane otimizado | Otimizado vs tradicional |
|---|---|---|---|---|
| 1 | 3,324 ms · 154 tok/s | 2,337 ms · 219 tok/s | **2,073 ms · 247 tok/s** | **1.60×** |
| 2 | 1,929 ms · 264 tok/s | 1,274 ms · 402 tok/s | **1,167 ms · 438 tok/s** | **1.65×** |
| 3 | 1,641 ms · 309 tok/s | 1,015 ms · 502 tok/s | **949 ms · 540 tok/s** | **1.73×** |
| 4 | 1,486 ms · 342 tok/s | 794 ms · 635 tok/s | **766 ms · 669 tok/s** | **1.94×** |
| 5 | 1,369 ms · 377 tok/s | 814 ms · 624 tok/s | **775 ms · 668 tok/s** | **1.77×** |
| 6 | 1,511 ms · 341 tok/s | 770 ms · 666 tok/s | **755 ms · 682 tok/s** | **2.00×** |

- **P1 (mais rápido em todo número de núcleos):** passou. Razão de tempo otimizado/tradicional, com IC95% < 1 em todas as
  18 células (6 núcleos × 3 seeds): 0,62 com 1 núcleo, 0,51 com 4, 0,49–0,50 com 6.
- **P2 (otimizações ≥ 10% sobre a V8 em 4 núcleos):** **falhou.** Ganho de 4–6% (razão 0,94–0,96). Com 1 núcleo o
  ganho é 11%; com mais núcleos a comunicação e o desbalanceamento pesam mais que o cálculo economizado.
- **P3 (qualidade preservada):** passou. Concordância de tokens otimizado × V8 = 100% em todas as células; acurácia
  do SeedPlane acima do tradicional nas 3 seeds (+0,5 a +0,9 pp).

## Escala (aceleração sobre 1 núcleo, L=1024)
Tradicional: 1,72 / 2,03 / 2,24 / 2,43 / **2,20** (piora com 6 núcleos). SeedPlane otimizado: 1,78 / 2,18 / 2,71 /
2,68 / 2,75. Como previsto no protocolo, o SeedPlane para de ganhar acima de 4 núcleos: 8 shards não se dividem
igualmente entre 5 ou 6 workers.

## L = 512
Diferenças menores e mistas: com 1 núcleo o otimizado é 8% mais rápido que o tradicional; entre 2 e 6 núcleos os
tempos ficam próximos (±10%), às vezes a favor do tradicional. Sequências curtas têm pouca atenção para economizar.

## Estrutura (do perfil `results/profile_v8.json`)
O ganho vem de duas fontes: (1) atenção local, que dá 1,6× com 1 núcleo; (2) paralelismo embaraçoso entre processos,
que escala melhor que as threads intra-operação do PyTorch (o tradicional satura em ~2,4×). Limite estrutural:
8 núcleos úteis por página de 1.024 tokens.

## Memória (`mem_bench.py`, `results/memory.json`, processo novo por medição, pico de RSS de toda a árvore de processos)
| Núcleos | Tradicional | SeedPlane otimizado | Razão |
|---|---|---|---|
| 1 | 400 MB | 696 MB | 1,7× |
| 2 | 403 MB | 1.027 MB | 2,5× |
| 4 | 403 MB | 1.688 MB | 4,2× |
| 6 | 407 MB | 2.343 MB | 5,8× |

A inferência em si acrescenta 0–6 MB nos dois métodos (o modelo é pequeno). A diferença vem de cada worker ser um
processo com sua própria cópia do PyTorch e dos pesos (~330 MB cada). Próxima otimização estrutural: workers como
threads compartilhando uma única cópia do modelo.
