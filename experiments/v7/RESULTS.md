# SeedPlane V7 — status (2026-09-23)

**Veredito: PENDENTE.** O modelo B (mensagens latentes) ainda está treinando; a avaliação pré-registrada roda em
seguida. Este arquivo será substituído pelo resultado (sem apagar este histórico do git).

## O que já aconteceu (tudo registrado em PROTOCOL.md antes de qualquer critério)
- Treino A #1 (6.000 passos): **falhou no gate de sanidade**. Consultas no acaso (CE 4,156; d=0 5,1%).
- Treino A #2 (curriculum de comprimento): **falhou**. Preenchimento não aprendido, consultas 20%.
- Diagnóstico: nenhum lr/clip resolve; causa = platô de otimização (sinal do token vizinho diluído na atenção).
  Correção: *token shift* (adendo 2).
- Treino A #3 (com token shift): diagnóstico exploratório em 32 sequências fora da avaliação: **100% das consultas em
  todas as distâncias** (decodificação global), preenchimento CE 0,147. O gate formal roda na avaliação.

Logs dos treinos que falharam: `results/run1_train_a.json`, `results/run2_train_a.json`.
