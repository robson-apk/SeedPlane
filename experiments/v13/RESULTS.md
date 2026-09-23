# V13 — resultados (em andamento)

## Correção (C1, C2) — Mac M4, CPU 2 threads, Qwen2.5-0.5B-Instruct F16 GGUF vs PyTorch fp32, WikiText-2, 2 trechos × 1.024 tokens

| trecho | NLL/token nativo, janela única | PyTorch completo | nativo, shards 512/256 | PyTorch shards | tokens pontuados |
|---|---|---|---|---|---|
| 0 | 2,2435 | 2,2356 | 2,6519 | 2,6339 | 1.023 em todas |
| 1 | 2,9867 | 2,9695 | 2,9822 | 2,9662 | 1.023 em todas |

- **C1 (worker vs PyTorch, janela única, limite de 1%): passou.** Diferença máxima de 0,58%, sempre para cima (F16 contra fp32).
- **C2 (shards, limite de 1%): passou.** Diferença máxima de 0,69%. O worker aplica as posições originais e o mesmo
  recorte de núcleo que o motor PyTorch; a contagem de tokens pontuados bate exatamente (1.023).
- **Pulado:** a parte de C1 que compara com o `llama-perplexity` (≤ 0,5%). O worker chama o próprio `llama_decode`,
  mas essa comparação **não foi medida**.
- Cobertura: 2 trechos, 1 dispositivo (CPU do Mac). Falta repetir no Vulkan/B580.

Medido com `experiments/v13/v13_correct.py`; JSON em `results/`.

## Velocidade (P1–P3)
Ainda não executada: ela depende do build Vulkan do worker no 5600X, que só será feito quando a medição da V12 terminar,
para não disputar CPU com ela.
