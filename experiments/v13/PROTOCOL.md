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
