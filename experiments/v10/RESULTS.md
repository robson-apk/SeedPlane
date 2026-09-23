# SeedPlane V10 — resultado (2026-09-23)

**H1 FALHOU · H2 FALHOU · H3 FALHOU · H4 PASSOU · Q PASSOU.** Critérios publicados no git (commit 2703278) antes da execução.

Endpoints: B580 (XPU) · 4 núcleos do Ryzen 5600X (1 processo/thread cada) · 2 núcleos do Mac M4 pela LAN (nice).
Mesmo checkpoint V6; distribuição proporcional à velocidade calibrada; sincronização a cada um dos 16 passos.

## Vazão — 32 páginas (média das 3 seeds)
| Dispositivos | SeedPlane | Tradicional (paralelismo de dados) | Razão |
|---|---|---|---|
| B580 | **11.710 tok/s** | 8.365 tok/s | 1,40× |
| B580 + 4 Ryzen | 10.628 tok/s | 8.366 tok/s | 1,27× |
| B580 + 4 Ryzen + 2 Mac | 9.090 tok/s | 7.155 tok/s | 1,27× |
| 4 Ryzen | 531 tok/s | 296 tok/s | 1,79× |
| 4 Ryzen + 2 Mac | **1.454 tok/s** | 719 tok/s | **2,02×** |

## Latência — 1 página de 1.024 tokens (mediana)
| Dispositivos | SeedPlane | Espera de sincronização |
|---|---|---|
| B580 | 199 ms (tradicional na B580: **93 ms**) | 3% |
| B580 + 4 Ryzen | 201 ms | 3% |
| B580 + 4 Ryzen + 2 Mac | 320 ms | 33% |
| 4 Ryzen | 705 ms | 1% |
| 4 Ryzen + 2 Mac | **480 ms** | 32% |

## Critérios
- **H1 (somar dispositivos reduz a latência em ≥10% frente à B580 sozinha): falhou.** Melhor conjunto: B580+4 Ryzen,
  201 ms contra 199 ms. Calibração: a B580 processa ~2.900 shards/s; um núcleo do Ryzen ~54/s; um núcleo do Mac ~200/s.
  Somar dispositivos 15–50× mais lentos só acrescenta espera de sincronização.
- **H2 (SeedPlane no melhor conjunto mais rápido que o tradicional na B580, 1 página): falhou.** Na GPU o tradicional
  é 2,1× mais rápido numa página só: a GPU não sofre com o custo quadrático em 1.024 tokens, e o SeedPlane vira muitos
  kernels pequenos.
- **H3 (vazão da B580+CPU+Mac ≥ 1,10 × B580 sozinha): falhou.** Caiu de 11.710 para 9.090 tok/s.
- **H4 (SeedPlane ≥ tradicional em vazão nos mesmos dispositivos): passou**, em todos os 5 conjuntos (1,27–2,02×).
- **Q (tokens idênticos entre dispositivos): passou.** Concordância de 100% com a B580 em todos os conjuntos.

## Achado descritivo (não era critério)
Clusters só de CPU escalam bem pela rede: 4 Ryzen → 4 Ryzen + 2 Mac aumentou a vazão 2,7× (531 → 1.454 tok/s, 2% de
espera) e reduziu a latência de uma página de 705 para 480 ms (32% de espera, dominada pela rede).

## Leitura
- **Onde o SeedPlane serve:** vazão em qualquer hardware (inclusive na GPU), e juntar CPUs comuns, inclusive de
  máquinas diferentes, para uma mesma página.
- **Onde não serve (hoje):** latência de uma página numa GPU, e misturar uma GPU rápida com CPUs lentas.
- Otimização aberta: os kernels do SeedPlane na GPU (muitos lotes pequenos por comprimento de janela) têm espaço para
  melhorar.
