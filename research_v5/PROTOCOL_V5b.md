# SeedPlane V5b — protocolo antes dos resultados (2026-09-22)

Motivo: o README publicado no commit 3e78abe afirma "3.5x wall-clock speedup" (41.9 → 11.95 ms, 1 → 4 workers, L=1024).
Esse número vem de `paired_runtime.py`, onde cada shard dorme `uniform(0, 5 ms)` (jitter injetado). Com 8 shards,
~20 ms de sleep são serializados no caso 1-worker e paralelizados no caso 4-workers. Além disso, o baseline é
"sharded com 1 worker", não um forward global sem shards.

Pergunta: sem jitter, a inferência sharded com 4 workers persistentes (incluindo IPC e fusão owner/halo com envelope
exato) é mais rápida que um único forward do mesmo checkpoint sobre a sequência inteira?

Comparadores (mesmo input por par, ordem randomizada dentro do par):
- `global_1t`: um forward completo no processo principal, `torch.set_num_threads(1)`.
- `global_4t`: idem, `torch.set_num_threads(4)` (mesmo orçamento de núcleos que 4 workers).
- `sharded_4w`: 4 workers persistentes, sem jitter, inferência + IPC + fusão.

L ∈ {512, 1024}; seeds 11/23/37; 20 pares por (seed, L). Warm-up antes de medir.

Critério (fixado antes de ver números): `sharded_4w` só tem "speedup" se reduzir o tempo médio em ≥10% frente ao
**melhor** dos dois globais, com IC95% bootstrap (2000 reamostragens de pares) com limite inferior > 0, nas três
seeds, em L=1024. Caso contrário, a alegação de speedup do README é falsificada neste hardware/modelo.

Escopo: timing apenas; mesmo checkpoint toy (D=32, 2 camadas), CPU local (Mac). Não mede qualidade. O checkpoint
global treinado não está no repositório; para timing a arquitetura é idêntica, então usa-se o checkpoint clmp.
