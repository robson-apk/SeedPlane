# SeedPlane V13 — SeedPlane dentro do llama.cpp (kernels otimizados) + distribuição GPU/CPU/outras máquinas
Protocolo antes dos resultados, 2026-09-23.

## Motivação
V12: nosso motor em PyTorch fp32 perde feio em velocidade bruta para o llama.cpp (Vulkan na B580: ~20.000 tok/s em
pp512). O SeedPlane é uma camada de escalonamento; os kernels devem ser os do llama.cpp.

## Implementação
- `native/seedplane-worker.cpp`: ferramenta nova sobre a API C do llama.cpp (código-fonte mais recente, compilado com
  Vulkan no 5600X e contra o llama.cpp do Homebrew no Mac). Carrega um GGUF no dispositivo escolhido (`--ngl 99` GPU,
  `--ngl 0` CPU com `-t` threads), recebe janelas por TCP em protocolo binário (sem pickle, com chave), processa cada
  janela com `llama_decode` usando **as posições originais dos tokens** (`llama_batch.pos`) e devolve NLL das posições
  do núcleo ou só o fim do prefill.
- Coordenador Python (`seedplane/llama_backend.py`) com o escalonador dinâmico com trava de cauda da V12.

## Validação de correção (antes de qualquer velocidade)
- **C1:** com uma janela só (S ≥ L), NLL do worker = NLL do `llama-perplexity`/forward completo do próprio llama.cpp
  (|Δ| ≤ 0,5% relativo) e próximo do PyTorch fp32 (|Δ| ≤ 1%, diferença F16).
- **C2:** com shards (S=512, H=256), NLL do worker vs `nll_shards` do PyTorch: |Δ| ≤ 1% relativo.

## Velocidade (processamento do prompt, Qwen2.5-0.5B-Instruct F16, 1 prompt)
L ∈ {4.096, 8.192, 16.384, 32.768}. Referência nativa: `llama-bench -p L` com o melhor build para a B580 (entre oficial
Vulkan/SYCL e os builds do usuário no SYNAPSE), e CPU com 6 threads.
SeedPlane-on-llama.cpp: {B580}, {B580 + CPU}, {B580 + CPU + Mac CPU 2 threads}, escalonador dinâmico.
- **P1:** em L=16.384, SeedPlane {melhor conjunto} processa o prompt mais rápido que o melhor llama.cpp nativo (3 reps).
- **P2:** somar dispositivos à B580 nunca piora mais que 3% (o escalonador deixa de fora quem atrasa).
- **P3 (qualidade no mesmo motor):** NLL SeedPlane / NLL janela única em L=16.384, WikiText, 3 trechos — reportado
  (sem limiar: a V12 já mostrou o custo; aqui é para acompanhar a velocidade).
Previsões: em 4.096 o nativo ganha (atenção ainda barata na GPU); em 16–32k o SeedPlane ganha (custo linear vs quadrático).
Escopo: um modelo pequeno; prefill/pontuação apenas (geração token a token depois).

## Adendo 1 (2026-09-23, antes de qualquer número de velocidade da V13) — escalar ao somar dispositivos
Diagnóstico com dados já medidos (V9, V10, V12):
- Só CPU, o ganho é real: de 1 para 4 núcleos, 2,7× (V9); de 4 Ryzen para 4 Ryzen + 2 Mac, 2,7× (V10).
- Com a B580, somar dispositivos **piorou**: de 11.710 para 10.628 e depois 9.090 tok/s (V10 H1/H3 falharam).
- Teto teórico: a B580 faz ~2.900 shards/s, e um núcleo do Ryzen ~54/s. Mesmo com overhead zero, somar 4 núcleos
  daria no máximo **+7%**. A perda de 9–22% vem do overhead: coordenador em Python com pickle, núcleos do Ryzen
  disputando a CPU que alimenta a GPU, e espera de sincronização.
- Conclusão: reescrever em C não aumenta o teto, só tira o overhead. O teto só sobe se cada núcleo ficar muito mais
  rápido (kernels do llama.cpp em vez de PyTorch fp32) ou se CPU e Mac fizerem outro tipo de trabalho (rascunho
  especulativo na V15, camadas do pipeline quando o modelo não cabe na V14).
Decisão: o caminho quente já é C++ (seedplane-worker). O coordenador só será reescrito em C++ se o overhead medido
passar de 5%.

Critérios novos (fixados agora):
- **P4 (monotonia):** em cada L, cada conjunto maior de dispositivos ≥ 0,97 × o melhor subconjunto dele.
- **P5 (eficiência de escala):** vazão do conjunto ≥ 0,80 × a soma das vazões individuais de cada dispositivo
  (medidas sozinhas, mesmo L e mesmo plano). Isso vale para o conjunto só de CPU (1, 2, 4, 6 workers no 5600X)
  e para o conjunto completo.
- **P6 (overhead):** fração do tempo total que não é compute_ms do worker, no conjunto completo. Se passar de 5%,
  reescrever o coordenador em C++ e medir de novo (V13b).
- Os núcleos dos workers de CPU devem deixar pelo menos 1 núcleo livre para alimentar a GPU. Isso será registrado
  por condição.
