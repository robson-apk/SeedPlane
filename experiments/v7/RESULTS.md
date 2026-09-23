# SeedPlane V7 — resultado (2026-09-23)

**Veredito pré-registrado: H1 FALHOU, H2 FALHOU** (C0 passou: o teste mediu o que devia).

Tarefa sintética de chave→valor em que a informação distante é necessária por construção. 3 seeds de avaliação ×
256 sequências; ~4.000–4.700 consultas com distância ≥ 2 shards por seed. Acaso = 1,6%.

| Consultas com a definição a ≥ 2 shards | seed 101 | seed 202 | seed 303 |
|---|---|---|---|
| **Atenção global (Transformer tradicional)** | **100%** | **100%** | **100%** |
| Shard isolado | 1,7% | 1,5% | 1,3% |
| **SeedPlane, halo de 16 tokens (desenho atual)** | 1,6% | 1,7% | 1,5% |
| SeedPlane v2, mensagens latentes, 8 rodadas | 1,5% | 1,3% | 1,4% |

- **C0 passou:** global 100% em todas as distâncias; isolado no acaso. Gate de sanidade (d=0 ≥ 90%): 100% nas 3 seeds.
- **H1 falhou (previsto e registrado antes):** recuperação R_tok = −0,001 / 0,002 / 0,002. O halo só transmite o que já
  está escrito como token perto da borda. Com o vizinho imediato (d=1), o halo acerta 13,6% (seed 101): exatamente os
  casos em que a definição cai dentro dos 16 tokens da borda.
- **H2 falhou:** o modelo B acerta 100% dentro do próprio shard, mas dá o mesmo resultado com 1, 2, 4 ou 8 rodadas,
  inclusive com d=1. **Ele não aprendeu a usar as mensagens.** Checagem de vazamento: passou (R=2, d≥4 = 1,2–1,4%).

## Leitura correta
- **Medido:** no SeedPlane atual, um shard não recebe informação que está a 2 ou mais shards de distância. Numa tarefa
  que depende disso, o Transformer tradicional vence com folga (100% contra acaso).
- **Não demonstrado:** que a troca latente entre vizinhos seja impossível. Esta implementação (1 vetor por direção,
  gradiente através de 8 rodadas, 9.000 passos) não aprendeu a usar o canal. Métodos de memória recorrente e message
  passing funcionam na literatura; fazer isto funcionar aqui exigiria um protocolo novo (V7b), com critérios novos.
  Este resultado não é reinterpretado.
- Consequência para o README: a vitória de qualidade do SeedPlane vale para texto em que o contexto relevante é
  local (TinyStories). Com dependência longa, hoje o SeedPlane perde.

Dados: `results/eval_rows.json` (por consulta), `results/analysis.json`, logs de treino `results/train_*.json`.
