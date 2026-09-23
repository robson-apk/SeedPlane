# SeedPlane V8 — head-to-head no MESMO modelo: velocidade E qualidade (protocolo antes dos resultados, 2026-09-23)

## Por que
O README mostra velocidade (V5b, modelo original pequeno) e qualidade (V6, modelo de 5,3M) medidas em modelos
diferentes. A V8 mede as duas coisas juntas, no mesmo checkpoint (`checkpoints/v6_mdlm_d256_l6_seed1.pt`), nas mesmas
execuções.

## Condições
- **Traditional:** um processo, atenção global na sequência inteira, 4 threads.
- **SeedPlane:** 4 workers persistentes (1 thread cada), cada um dono de 2 shards de 128 tokens com halo de 16;
  a cada passo recebem o estado atual e devolvem só (token previsto, confiança) das suas posições.
Mesma regra de commit (V6: cota por shard, argmax), K=16 passos, 50% mascarado. CPU do 5600X, sem outros jobs
pesados rodando. L ∈ {512, 1024}. Ordem das condições sorteada por par.

## Métricas (todas nas mesmas sequências)
- Tempo total da decodificação iterativa (inclui IPC e fusão no SeedPlane).
- Qualidade: acurácia final do preenchimento; NLL de um passo nos tokens mascarados com 15%, 50% e 90% de máscara.
3 seeds (11, 23, 37) × 32 sequências por L. IC95% por bootstrap pareado sobre sequências.

## Critérios (fixados agora)
- **Q — qualidade não-inferior:** em L=1024, nas 3 seeds, acurácia SeedPlane − Traditional com limite inferior do IC
  > −0,5 pp, E NLL SeedPlane ≤ 1,01 × NLL Traditional nas três taxas de máscara.
- **S — velocidade:** em L=1024, tempo SeedPlane ≤ 0,90 × Traditional (IC superior da razão < 0,90) nas 3 seeds.
- Veredito "mesma qualidade (ou melhor) E mais rápido" só se Q e S passarem. Qualquer outra combinação é reportada
  como saiu, inclusive no README.
Escopo: TinyStories (pouca dependência longa; o teste de contexto distante é a V7), CPU, modelo de 5,3M.
