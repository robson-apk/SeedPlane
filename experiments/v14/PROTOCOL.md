# SeedPlane V14 — pipeline EXATO entre dispositivos heterogêneos, com divisão automática de camadas
Protocolo antes dos resultados, 2026-09-23.

## Por que (revisão de arquitetura)
As janelas SeedPlane são aproximadas (V12: +3,7% a +13% de perplexidade no Qwen sem retreino; V7: perdem informação
distante). Para usar vários dispositivos SEM perda num único prompt, a arquitetura conhecida é pipeline por camadas:
cada dispositivo guarda uma faixa de camadas e o prompt passa em micro-lotes; só ativações cruzam a rede. O llama.cpp
já faz isso via RPC (`--rpc`, `-ts`), mas a divisão (`--tensor-split`) é manual. A proposta do SeedPlane passa a ser um
**planejador**: mede os dispositivos e escolhe a divisão (e, depois, a estratégia).

## Configuração
Qwen2.5-0.5B-Instruct F16 GGUF. llama.cpp oficial b11140 (Vulkan) no 5600X + `ggml-rpc-server` em:
B580 (Vulkan), CPU do 5600X, e Mac (CPU, 2 threads, llama.cpp compilado do fonte com GGML_RPC).
Métrica: `llama-bench -p L -n 0` (tok/s de prompt) e `-n 128` (tok/s de geração), L ∈ {4.096, 16.384}, 3 repetições.

## Condições
1. B580 sozinha (referência).
2. B580 + CPU local via RPC, divisão manual padrão (`-ts` proporcional à memória, o default).
3. B580 + CPU local, divisão automática do planejador (proporcional ao tempo por camada medido em cada dispositivo).
4. B580 + CPU + Mac, divisão automática.

## Critérios (fixados agora)
- **E1 (exatidão):** saída idêntica: mesmos tokens gerados (greedy, 64 tokens) em 1 e em 4 (texto igual).
- **E2 (planejador vs manual):** tok/s de prompt da condição 3 ≥ 1,05 × condição 2.
- **E3 (somar dispositivos):** melhor condição multi-dispositivo ≥ 1,0 × B580 sozinha em prompt (não piorar).
Previsão registrada: com um modelo de 0,5B a B580 é tão mais rápida que somar CPU/Mac provavelmente NÃO acelera
(E3 pode falhar; o custo de rede e de sincronização domina). O ganho de pipeline aparece quando o modelo não cabe
num dispositivo só ou quando os dispositivos têm velocidades comparáveis.
