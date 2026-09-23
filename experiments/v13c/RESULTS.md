# V13c: fronteira qualidade × velocidade (B580, Qwen2.5-0.5B F16, 16.384 tokens, WikiText-2 trechos 3–5, novos)

Razão de perplexidade = exp(NLL − NLL da atenção completa), nos mesmos trechos. Velocidade: prefill, mediana de 3.
llama.cpp nativo em 16.384 = 3.891 tok/s (V13). Brutos: `results/frontier_1.json`
(`frontier.json` = execução 1, parcial, parou no bug de posições; ver adendo).

| modo | bloco / S | halo H | sumidouros | trecho 3 | trecho 4 | trecho 5 | **média** | tok/s | × nativo |
|---|---|---|---|---|---|---|---|---|---|
| janelas | 512 | 256 | 0 | 1,160 | 1,200 | 1,160 | 1,174 | 12.320 | 3,17 |
| janelas | 512 | 256 | 4 | 1,160 | 1,195 | 1,178 | 1,178 | 9.416 | 2,42 |
| janelas | 1.024 | 1.024 | 4 | 1,055 | 1,062 | 1,060 | 1,059 | 7.632 | 1,96 |
| janelas | 2.048 | 2.048 | 4 | 1,015 | 1,014 | 1,031 | 1,020 | 5.814 | 1,49 |
| janelas | 4.096 | 4.096 | 4 | 1,003 | 1,001 | 1,009 | 1,004 | 4.100 | 1,05 |
| span | 512 | 256 | 4 | 1,193 | 1,233 | 1,201 | 1,209 | 19.399 | 4,99 |
| span | 512 | 1.024 | 4 | 1,061 | 1,078 | 1,056 | 1,065 | 14.136 | 3,63 |
| **span** | **512** | **2.048** | **4** | **1,012** | **1,012** | **1,011** | **1,012** | **10.488** | **2,70** |
| **span** | **512** | **4.096** | **4** | **1,000** | **0,998** | **0,997** | **0,998** | **7.175** | **1,84** |
| span | 1.024 | 8.192 | 4 | 0,999 | 0,999 | 0,999 | 0,999 | 4.758 | 1,22 |

## Veredito
- **F1 (≤ 1,02 de perplexidade E ≥ 2× o nativo): PASSOU.** Span 512/2.048: 1,012 com 2,70×. O mesmo resultado nos 3 trechos.
- **F2 (≤ 1,05 E ≥ 3×): FALHOU.** O ponto mais próximo é o span 512/1.024: 1,065 com 3,63×.
- **Qualidade igual à da atenção completa:** span 512/4.096 dá 0,998 (−0,2%) com 1,84× o nativo; o span
  1.024/8.192 dá 0,999. A diferença para o modelo original está dentro do ruído de 3 trechos: é **igual, não
  melhor**.
- O span domina as janelas em toda a fronteira: com a mesma qualidade é 1,7–1,8× mais rápido, porque não recalcula o halo.
- Anomalia registrada: nas janelas, 4 sumidouros custaram 24% de velocidade (12.320 → 9.416), sem ganho de
  qualidade. Isto não foi investigado ainda.
- Cobertura: um modelo (0,5B), um corpus (WikiText), um comprimento (16k), 3 trechos. Custo do span em 32k não
  medido aqui; será medido na V13d.
