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

## Adendo 1 (2026-09-23, antes de qualquer medida)
- Build: llama.cpp 6e60f35 (Vulkan + RPC) nos dois lados, em vez do oficial b11140, porque o protocolo RPC tem de
  ser a mesma versão no Mac. `ggml-rpc-server` da CPU local com 5 threads (1 núcleo fica para alimentar a GPU);
  Mac com 2 threads, `nice`.
- Ordem de dispositivos: Vulkan0 (B580), RPC0 (CPU 5600X), RPC1 (Mac).
- Entrada do planejador: pp512 de cada dispositivo sozinho (`llama-bench -dev X`), medida no início (pré-bench).
- Pela filosofia do autor (nenhum dispositivo fica de fora), cada condição multi-dispositivo roda em 3 divisões:
  padrão do llama.cpp (proporcional à memória livre), planejador (pode dar 0 camadas) e planejador-min1 (≥ 1 camada
  por dispositivo). E2 compara planejador × padrão. E1 usa a condição de 3 dispositivos com min1, para que todos
  participem de fato.
- Métricas: `llama-bench -p 4096,16384 -n 128 -r 3` (prompt e geração).

## Adendo 2: execução 1 abortada (`results/v14.json` guardado, só com o pré-bench e a divisão padrão de 2 dispositivos)
Com `-dev Vulkan0,RPC0 -ts 24/0` (o planejador mandou tudo para a B580), o servidor RPC da CPU continuou a 500% de
CPU: o llama.cpp ainda manda trabalho para um dispositivo listado com 0 camadas. Correção da ferramenta: só entram
em `-dev` os dispositivos com > 0 camadas. Critérios inalterados. Números já vistos: pré-bench (B580 20.525, CPU via
RPC 270, Mac via RPC 210 tok/s) e padrão B580+CPU (pp4096 221, pp16384 143, tg128 34 tok/s). Eles serão medidos de
novo na execução 2, e só a execução 2 conta.

## V14b: a ordem importa? (pré-registrado 2026-09-23, antes de medir; motivado pela V14: 23/1 = 222 tok/s ≈ padrão)
Hipótese: o llama.cpp põe a cabeça de saída (896 × 151.936, a maior multiplicação do modelo) no dispositivo da ÚLTIMA
camada. Com `-dev Vulkan0,RPC0 -ts 23/1`, a cabeça fica na CPU via RPC. Ao inverter a ordem (CPU primeiro), a cabeça
fica na B580.
Condições (`llama-bench -p 4096,16384 -n 128 -r 3`): `-dev RPC0,Vulkan0 -ts 1/23` e `-dev RPC1,RPC0,Vulkan0 -ts 1/1/22`.
Também registrado: onde o llama.cpp aloca a camada de saída (log `load_tensors`).
- **O1:** CPU primeiro com 1/23 ≥ 5 × a mesma divisão com a CPU por último (≥ 1.111 tok/s em pp4096).
- **O2:** CPU primeiro com 1/23 ≥ 0,5 × B580 sozinha em pp4096 (≥ ~5.400 tok/s).
- **O3:** a cabeça de saída está no último dispositivo da lista (verificado no log).
Previsão: O1 e O3 passam; O2 é incerto (transferência de ativações por RPC a cada micro-lote).
