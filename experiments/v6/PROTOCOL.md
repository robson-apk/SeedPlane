# SeedPlane V6 — difusão iterativa com shards: protocolo antes dos resultados (2026-09-22)

## Pergunta
Numa difusão mascarada iterativa (K passos), shards de 128 tokens que trocam **só as bordas (halo) com vizinhos a cada
passo** chegam perto da decodificação global? O erro nas fronteiras fica limitado quando o número de shards cresce, ou
se propaga? E — o ponto que tornaria a ideia relevante — a troca iterativa entre vizinhos recupera informação de
**longo alcance** (fora da janela do shard), ou só informação local?

## Modelo (treinado do zero, B580, uma vez; não se escolhe checkpoint pelo teste)
Transformer encoder bidirecional, d=256, 6 camadas, 8 cabeças, FF=1024, vocab 1024 (cache word-level do projeto),
posições absolutas até 1024. Objetivo de difusão mascarada: taxa de máscara ~U(0.05, 1.0), CE nos mascarados.
Treino misto 50/50: sequências completas L=1024 e janelas locais (128 + 2H, H∈{16,64}) com posição absoluta.
Um único modelo serve a todos os decodificadores — a única diferença entre eles é o contexto visível.
10.000 passos, AdamW 3e-4, warmup 500, cosseno. `model.eval()` em toda avaliação.

## Decodificadores (mesmas regras de commit)
Tarefa: preencher texto de validação com taxa de máscara r∈{0.5, 0.9}, K∈{1, 16} passos. Em cada passo cada shard
commita os ceil(restantes/(passos restantes)) tokens mascarados mais confiantes do seu núcleo (quota por shard, igual
para todos os decodificadores). Greedy (argmax).
- `global`: vê a sequência inteira no estado atual.
- `halo16` / `halo64`: shard vê núcleo + H tokens de cada vizinho, no estado do passo anterior (só vizinhos comunicam).
- `isolated`: shard vê só o próprio núcleo (sem comunicação).

## Conjuntos avaliados
Seeds 11/23/37 (sequências e máscaras diferentes do split de validação), 64 sequências por seed.
L ∈ {256, 512, 1024} → S = 2, 4, 8 shards.
Subconjuntos: `boundary` (≤16 tokens de um corte interno), `interior`, e `LR` (longo alcance): token mascarado cujo
valor verdadeiro não está entre os 50 mais frequentes, aparece visível na sequência **fora** da janela do shard±16 e
**não** aparece visível dentro dela. Cobertura (n) reportada em cada célula.

## Critérios (fixados agora)
Métrica: acurácia final sobre os mascarados. IC95% por bootstrap sobre sequências (2000 reamostragens).
Referência principal: L=1024, r=0.5, K=16.

- **C0 — manipulação exercida.** `global − isolated` no LR ≥ 2 pp com IC excluindo 0 em todas as seeds.
  Se falhar: INCONCLUSIVO (o modelo não usa contexto longo; a pergunta não foi testada).
- **H1 — erro de fronteira limitado.**
  (a) `global − halo16` em todos os mascarados ≤ 1 pp (limite superior do IC) nas três seeds;
  (b) o gap na fronteira não cresce com S: gap_boundary(S=8) − gap_boundary(S=2) ≤ 1 pp;
  (c) passos não amplificam: gap_all(K=16) − gap_all(K=1) ≤ 0,5 pp.
- **H2 — recuperação de longo alcance (a parte que seria nova).**
  Recuperação R = (halo16 − isolated)/(global − isolated) no LR ≥ 0,5 nas três seeds.

Veredito: H1 e H2 passam → mecanismo candidato a contribuição (ainda em escala pequena, exige replicação maior).
Só H1 → "funciona como atenção local": resultado esperado, sem novidade. C0 falha → inconclusivo.
Não reajustar limiares, seeds ou passos depois de ver números. Resultado negativo é registrado como negativo.

## Escopo
Modelo pequeno (~5M), vocabulário de palavras, TinyStories (dependência de longo alcance fraca por natureza),
single-GPU simulando a troca entre shards (a comunicação é semântica, não medida em rede). Não mede velocidade.

## Adendo antes de qualquer resultado (mesmo dia)
Só ~3 GB de VRAM estavam livres na B580 (algo externo ocupa ~8 GB; WMI travado impediu identificar). Batch efetivo
mantido (8 sequências completas / 48 janelas) via 2 micro-batches com acumulação de gradiente, autocast bf16 no
forward. Decodificação global em blocos de 8 sequências (não altera o resultado). Nenhum número de avaliação existia
quando esta mudança foi feita; o smoke test local usou modelo minúsculo não treinado e foi descartado.

## Adendo 2 — treino 1 falhou, antes de qualquer avaliação dos critérios (2026-09-23)
O treino 1 (10k passos, lr 3e-4, logits × 1/√d, embeddings N(0,1) — herdado do checkpoint original) colapsou para o
unigrama: loss global mascarada ≈ 4,96–4,98 para taxas de máscara 0,15–0,9 (unigrama 5,10); logits quase constantes
entre posições. Verificado: XPU = CPU (|Δ| 5e-6), não é bug de kernel. Teste de arquitetura minúsculo em CPU
(2.500 passos): atual 4,99 vs escala 1 + init 0,02 + lr 1e-3 → 2,53. Nenhum critério (C0/H1/H2) foi calculado com o
modelo 1; ele é arquivado como falha de treino (`results/run1_*`).
Correção: logits sem escala, embeddings N(0, 0,02), lr 1e-3. Resto igual.
**Gate de sanidade fixado agora:** antes de avaliar, loss global com máscara 0,15 ≤ unigrama − 1,0 nats. Se falhar,
parar e reportar falha de treino — não rodar os critérios.

## Adendo 3 — treino 2 também falhou no gate de sanidade; causa isolada (2026-09-23)
Treino 2 (escala/init corrigidas): loss global com máscara 0,15 = 4,92 (gate exigia ≤ 4,10) → falha registrada,
critérios não calculados. Diagnóstico (1.500 passos, d=256, 6 camadas, máscara 0,15, L=256):
fp32 sem clip 3,07 · bf16 sem clip 3,30 · bf16 + clip 4,95 · fp32 + clip 4,70. Causa: `clip_grad_norm_(1.0)`.
Correção: sem clipping. Resto igual. Gate de sanidade inalterado. Checkpoints dos treinos 1 e 2 arquivados.

## Adendo 4 — correção do adendo 3 + curriculum (2026-09-23, antes de qualquer avaliação dos critérios)
**Correção:** o adendo 3 atribuiu o colapso ao clip com base em UMA rodada por configuração. Repetições mostram que
sair do platô do unigrama é estocástico (a XPU não é determinística): d=256/6L, bf16, sem clip, máscara 0,15 →
escapou em 1 de 2 seeds; na repetição da mesma configuração do adendo 3, não escapou (4,98 no passo 2500).
d=128/4L → 2/2; embeddings N(0,1) → 0/2. O treino 3 (sem clip, máscara U(0,05; 1)) falhou no gate (4,96).
A causa do clip NÃO está demonstrada.
**Procedimento novo:** curriculum de máscara — passos 0–2999 com taxa U(0,05; 0,3), depois U(0,05; 1,0).
Arquitetura (d=256, 6 camadas) mantida. Seed de treino variável: tentar seeds 1, 2, 3 nessa ordem e usar a primeira
que passar no gate de sanidade (loss@0,15 ≤ unigrama − 1,0). Isso seleciona pelo treino, não pelo teste. Se as três
falharem: reportar falha de treino e parar. Tentativas registradas em results/.
