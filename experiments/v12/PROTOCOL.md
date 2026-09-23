# SeedPlane V12 — Fase 1: modelos autoregressivos existentes SEM retreino (Qwen2.5-0.5B)
Protocolo antes dos resultados, 2026-09-23.

## Objetivo
Levar a ideia do SeedPlane (atenção em shards com halo, distribuída entre dispositivos) a um modelo pré-treinado que NÃO
foi treinado assim, medindo quanto de qualidade se perde e quanto de velocidade se ganha — contra o próprio modelo e
contra o llama.cpp nativo. (Fase 2, depois: adaptar o modelo por treino, como Dream/DiffuLLaMA fizeram.)

## Modelo e dados
`Qwen/Qwen2.5-0.5B-Instruct` (safetensors, sem alterar pesos). Texto: TinyStories (split de validação) e um corpus geral
(WikiText-2 test, se disponível; senão só TinyStories — registrado no resultado).

## Variantes de atenção (causal, zero-shot)
- `full`: atenção causal completa (o modelo original).
- `sp(S,H)`: cada token só vê o próprio shard de S tokens + H tokens do shard anterior (bloco causal com halo).
  S ∈ {256, 512}, H ∈ {64, 256}.
- `sp_sink(S,H)`: igual, mais os 4 primeiros tokens da sequência sempre visíveis ("attention sink", truque conhecido
  para atenção local sem retreino).

## Qualidade
Perplexidade em L ∈ {1.024, 2.048, 4.096}; 3 seeds × 8 sequências por L e corpus; `model.eval()`.
- **Q1:** existe variante `sp*` com perplexidade ≤ 1,05 × `full` em L=4.096 nos dois corpora, 3 seeds.
- Previsão registrada: `sp` puro perde mais que 5%; `sp_sink` com S=512 é o melhor candidato.

## Velocidade (processamento do prompt, L=4.096, 1 sequência)
Nosso motor: `full` e a melhor variante `sp*` em {1, 2, 4, 6 núcleos do Ryzen}, {B580}, {B580 + CPU}, {CPU + Mac}.
llama.cpp nativo: `llama-bench -p 4096` com o mesmo modelo em GGUF F16 (mesma classe de precisão), 6 threads no Ryzen
e no Mac; e a GPU se houver build com SYCL/Vulkan.
- **S1:** o melhor `sp*` no melhor conjunto de dispositivos processa o prompt mais rápido (tokens/s) que o llama.cpp no
  melhor dispositivo único, 3 repetições.
- **S2:** `sp*` mais rápido que `full` no nosso motor, com o mesmo número de núcleos (ganho algorítmico isolado).
- Previsão registrada: S2 provável; S1 incerto — o llama.cpp é C++ altamente otimizado.
Escopo: 1 modelo pequeno; sem geração token a token nesta fase (medida separada depois); sem quantização.
