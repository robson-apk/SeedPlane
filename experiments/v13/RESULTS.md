# V13 — resultados: janelas SeedPlane sobre os kernels do llama.cpp

Qwen2.5-0.5B-Instruct F16 GGUF · 5600X + Intel Arc B580 (Vulkan) + Mac M4 (2 threads, LAN) · WikiText-2 · S=512, H=256.
Nativo = `llama-bench -p L` (build Vulkan atual, padrão). Nosso = `seedplane-worker` (mesmo llama.cpp, sem OpenMP) +
coordenador Python. Mediana de 3. Brutos: `results/speed_run2.json`. A execução 1 foi **abortada e é inválida**
(`speed_run1_ABORTED_invalid.json`, ver adendo 3).

## Correção
- **C1 e C2: passaram** (Mac CPU; |Δ NLL| 0,58% e 0,69% contra o PyTorch fp32). A comparação com `llama-perplexity` foi pulada.

## Velocidade (tok/s de prompt, só B580)

| L | llama.cpp nativo | SeedPlane janelas | SeedPlane span | span / nativo |
|---|---|---|---|---|
| 4.096 | 10.827 | 12.804 | 19.533 | 1,80× |
| 8.192 | 6.817 | 12.483 | 19.434 | 2,85× |
| 16.384 | 3.891 | 12.336 | 19.413 | 4,99× |
| 32.768 | 2.096 | 12.263 | 19.394 | 9,25× |

CPU nativo, 6 threads, `-nopo 1`: 225 (4.096) e 190 (8.192) tok/s; L maior foi pulado.

## Qualidade (mesmo motor, B580, L=16.384, 3 trechos, 16.383 tokens pontuados cada)

| trecho | NLL completo | janelas | span | perplexidade janelas/completo | span/completo |
|---|---|---|---|---|---|
| 0 | 2,399 | 2,618 | 2,653 | 1,245 | 1,290 |
| 1 | 2,584 | 2,755 | 2,790 | 1,186 | 1,229 |
| 2 | 2,591 | 2,745 | 2,777 | 1,167 | 1,204 |

**O ganho de velocidade tem preço: perplexidade +17% a +25% (janelas) e +20% a +29% (span) em 16k tokens**, no
Qwen sem retreino. Em 4k (V12) o custo era +13%. Quanto mais longo o texto, mais contexto distante as janelas cortam.

## Memória (buffers de execução do próprio llama.cpp, além dos pesos; B580)

| L | KV completo | KV SeedPlane | compute (ambos) | total completo | total SeedPlane | razão |
|---|---|---|---|---|---|---|
| 8.192 | 99 MiB | 12 MiB | 300 MiB (+12 / +5 host) | 411 | 317 | 0,77 |
| 16.384 | 195 MiB | 12 MiB | 300 MiB | 515 | 317 | 0,62 |
| 32.768 | 387 MiB | 12 MiB | 300 MiB (+36 / +5 host) | 723 | 317 | 0,44 |

O KV do SeedPlane é constante (**32× menor em 32k**), mas o buffer de
compute (~300 MiB, dominado pelos logits de um micro-lote de 512 × 151.936 do vocabulário) é igual nos dois e domina
o total. CPU, 8.192: 312 contra 399 MiB (0,78).

## Somar dispositivos (escalonador por pedaço, nenhum dispositivo descartado por política)

| L | só GPU (span) | + CPU 4 | + CPU 6 | + CPU 4 + Mac | + CPU 6 + Mac |
|---|---|---|---|---|---|
| 4.096 | 19.533 | 19.517 | 19.530 | **17.986** | **17.998** |
| 8.192 | 19.434 | 19.442 | 19.452 | 19.203 | 19.139 |
| 16.384 | 19.413 | 19.401 | 19.415 | **19.994** | **19.954** |
| 32.768 | 19.394 | 19.400 | 19.394 | **20.196** | **19.616** |

Quem recebeu tokens: o Mac sempre (44 a 2.048 tokens). Os slots de CPU do 5600X quase nunca: rodando junto com a GPU,
cada slot cai para ~90–113 tok/s, e o planejador não achou pedaço que terminasse a tempo.
Só CPU (janelas, L=4.096): 1, 2 e 3 slots × 2 threads = 92, 137 e 154 tok/s.

## Veredito (critérios pré-registrados)
- **P1 (SeedPlane em 16k mais rápido que o melhor nativo): PASSOU.** 19.994 contra 3.891 tok/s (5,1×). Mesmo só com a GPU: 4,99×.
- **P2 (somar dispositivos não piora mais de 3%): FALHOU em 4.096** (Mac: −7,9% no span, −5,5% nas janelas). Passou em ≥ 8.192.
- **P3 (qualidade, reportada sem limiar):** perplexidade +17–25% (janelas) e +20–29% (span) em 16k. Custo grande.
- **P4 (monotonia ≥ 0,97): FALHOU em 4.096** (a mesma queda com o Mac). Passou em ≥ 8.192.
- **P5 (eficiência de escala ≥ 0,80, só CPU): FALHOU.** 137 / (2 × 92) = 0,74 e 154 / (3 × 92) = 0,56. O campo
  `P5_efficiency` do JSON usa a vazão de uma janela isolada (tokens de halo incluídos) como base; a definição
  pré-registrada é "mesmo L e mesmo plano", ou seja, 92 tok/s. Aqui vale esta tabela.
- **P6 (overhead do coordenador ≤ 5%): PASSOU** (0,1–0,3%). Não é preciso reescrever o coordenador em C++.
- **P7 (em ≥ 16k, todo dispositivo > 0 tokens e vazão ≥ B580 sozinha): FALHOU** na primeira parte (os slots de CPU
  ficaram com 0 tokens). A vazão passou (+3,0% em 16k, +4,1% em 32k).
- **P8 (medido ≥ 0,90 × previsto): PASSOU em todas as linhas.** O medido ficou entre 0,97× e 1,21× do previsto.
- **P9 (span ≥ 1,3 × janelas, só GPU, L ≥ 8.192): PASSOU** (1,56–1,58×).
- **C3 (NLL span ≤ 1,01 × janelas): FALHOU** por pouco: 1,0134 / 1,0130 / 1,0116 (média 1,0127).
- **M1 (memória de execução ≤ 0,5 × completo em L ≥ 8.192): FALHOU em 8k (0,77) e 16k (0,62), passou em 32k (0,44).**

## Leitura
1. Os kernels do llama.cpp mais as janelas deram o maior ganho de velocidade medido até aqui: 5× em 16k e 9× em
   32k sobre o llama.cpp nativo, com memória de KV constante.
2. O preço é qualidade: sem retreino, cortar o contexto custa 17–29% de perplexidade em 16k. **Isto não é "mesma
   qualidade".** O próximo passo é a fronteira velocidade × qualidade (S e H maiores, sumidouros globais), antes de
   qualquer afirmação de uso geral.
3. A CPU ao lado de uma GPU 74× mais rápida contribui ~0 num prompt único. O Mac contribui +3–4% em textos longos e
   atrapalha em textos curtos: a rede custa mais que o pedaço. O planejador precisa incluir o custo de rede
   (o P8 mostra que ele subestima o Mac em 4k).
4. Memória: o termo que cresce com L (KV) ficou constante e 32× menor em 32k. O termo fixo (logits do micro-lote)
   agora domina e pode ser reduzido (micro-lote menor ou logits só onde precisa).
