# V13c: fronteira qualidade × velocidade (protocolo antes dos resultados, 2026-09-23)

Motivação: a V13 deu 5–9× sobre o llama.cpp nativo, com perplexidade +17–29% em 16k. O autor exige qualidade igual ou
melhor. Pergunta: existe uma configuração, SEM retreino, que perca ≤ 2% e ainda seja ≥ 2× mais rápida que o nativo?

Dados: WikiText-2, trechos 3, 4 e 5 de 16.384 tokens (NOVOS; a V13 usou 0–2). B580, Vulkan, mesmo worker.
Referência de qualidade: NLL da atenção completa (janela única) nos mesmos trechos. Referência de velocidade:
llama.cpp nativo em 16.384 = 3.891 tok/s (V13, medido).

Configurações (todas com 4 sumidouros globais ou de trecho, exceto a linha de base da V13):
- janelas (S, H, sinks): (512, 256, 0) [base V13], (512, 256, 4), (1024, 1024, 4), (2048, 2048, 4), (4096, 4096, 4)
- span (bloco, keep, sinks): (512, 256, 4) [V13], (512, 1024, 4), (512, 2048, 4), (512, 4096, 4), (1024, 8192, 4)
Métricas: razão de perplexidade exp(NLL − NLL_completo) por trecho e média; tok/s (prefill, mediana de 3, 16.384 tokens).

Critérios:
- **F1:** existe configuração com razão de perplexidade média ≤ 1,02 E velocidade ≥ 2 × nativo (≥ 7.782 tok/s).
  Se nenhuma existir, a afirmação "mesma qualidade" fica falsificada para este modelo sem retreino.
- **F2:** existe configuração com razão ≤ 1,05 E ≥ 3 × nativo (≥ 11.673 tok/s).
- Reportado: a fronteira de Pareto completa, sem escolher só o melhor ponto.
