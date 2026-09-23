# V12 — velocidade (processamento do prompt, L=4.096, Qwen2.5-0.5B-Instruct, WikiText-2, plano 512/256/0)

5600X + B580. Mediana de 3 repetições. Nosso motor: PyTorch fp32 (CPU) / XPU. llama.cpp: F16 GGUF, `llama-bench -p 4096`.
Execução em fases (adendo 2). JSON bruto: `results/speed.json`.

| Dispositivo | llama.cpp nativo | Nosso motor, atenção completa | Nosso motor, janelas SeedPlane |
|---|---|---|---|
| CPU 1 thread/worker | 71 | 71 | 92 |
| CPU 2 | 138 | 118 | 159 (fixo) · 142 (dinâmico) |
| CPU 4 | 254 | 143 | pulado (RAM) |
| CPU 6 | 318 | 141 | pulado (RAM) |
| B580 | **10.815** (Vulkan oficial) · 10.815 (Vulkan SYNAPSE) · 10.684 (SYCL oficial) · 9.058 (SYCL SYNAPSE) | 5.897 | 8.763 |
| Combinações GPU+CPU(+Mac) | — | — | puladas (RAM) |

Build `SYNAPSE build_dnn` (SYCL+oneDNN): **falhou** com
`GGML_ASSERT(ptr == pool_addr + pool_used)` em `ggml-sycl.cpp:1839`; não há número dele.

## Veredito
- **S1 (nosso melhor ≥ melhor llama.cpp): FALHOU.** 8.763 contra 10.815 tok/s (0,81×). Cobertura: as combinações de
  dispositivos não foram medidas (RAM). Na V10 elas foram mais lentas que a B580 sozinha, então é improvável que
  mudassem o veredito (inferência, não medição).
- **S2 (janelas > atenção completa, mesmo motor): PASSOU nas 3 células medidas.** 1,30× (CPU 1), 1,35× (CPU 2),
  1,49× (B580). CPU 4 e CPU 6 não foram medidas.
- **S3 (somar dispositivos não piora mais de 3%): NÃO EXECUTADO.** Todas as combinações foram puladas por RAM.
  Na V10, com o mesmo tipo de worker, H3 falhou (piorou).

## O que isso diz
- O ganho algorítmico das janelas existe (S2). Mas o motor em PyTorch perde para os kernels do llama.cpp, então o
  ganho algorítmico não compensa a diferença de implementação.
- Escala por threads: o llama.cpp na CPU vai de 71 para 318 tok/s (4,5× com 6 threads); a nossa atenção completa em
  PyTorch satura em ~2×. Isso motiva a V13: janelas SeedPlane sobre os kernels do llama.cpp.
