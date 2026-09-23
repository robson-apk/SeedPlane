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

## Adendo 1 — escalonador e builds do llama.cpp (antes de qualquer medida de velocidade)
Resultado de qualidade já saiu (Q1 falhou; melhor variante sp S=512 H=256 sinks=0) — a velocidade usa essa variante.
- **Escalonador dinâmico** (`cli.distribute_dynamic`): fila por demanda (cada worker pede a próxima janela ao terminar),
  tempo de ida-e-volta por worker medido continuamente (rede incluída) e trava de cauda (só entrega uma janela a um worker
  se ele a termina antes do resto do grupo terminar a fila sem ele). Motivo: na V10 o escalonador proporcional estático
  ignorava custo fixo por mensagem, subestimava a GPU (calibração com lote pequeno) e não sabia deixar um dispositivo de fora.
  Medido lado a lado com o estático.
- **S3 (escalonador):** com o dinâmico, {B580 + 4 Ryzen} e {B580 + 4 Ryzen + 2 Mac} nunca ficam mais de 3% abaixo de {B580}
  sozinha (somar dispositivos não pode piorar).
- **llama.cpp:** além dos binários oficiais (CPU, SYCL, Vulkan b11140), os builds do próprio usuário para a B580
  (F:\S.Y.N.A.P.S.E\llama.cpp: SYCL icx F16, SYCL+oneDNN, Vulkan). O melhor deles é a referência de S1.

## Adendo 2 (2026-09-23, antes de qualquer número de velocidade) — execução em fases
A 1ª execução de velocidade morreu sem resultado: em `cpu4`, os 4 workers fp32 (~2,3 GB cada) + o processo
coordenador (3,6 GB retidos dos modelos da fase "full") esgotaram os 16 GB do 5600X; os workers morreram por falta de
memória e o coordenador ficou esperando. Nenhum número foi salvo nem visto. Mudança: as fases (full / seedplane /
llama.cpp) rodam em processos separados e salvam após cada configuração; configurações sem RAM livre suficiente são
registradas como `skipped` (não executadas), nunca como resultado. Critérios S1–S3 inalterados.
