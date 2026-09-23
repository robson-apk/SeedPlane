# SeedPlane V7 — dependência longa sintética: protocolo antes dos resultados (2026-09-23)

## Por que
A V6 foi inconclusiva: no TinyStories quase não há informação distante útil (C0 falhou). A V7 cria uma tarefa em
que a informação distante é **necessária por construção** e pergunta:
*shards que só conversam com vizinhos conseguem recuperar informação que está a vários shards de distância?*

## Tarefa (gerador determinístico por seed)
Sequência de L=1024 tokens = 8 shards de 128. Vocabulário 1024: 0 PAD, 1 MASK, 2 DEF, 3 QRY, chaves 10–73 (64),
valores 100–163 (64), preenchimento 200–1023 gerado por uma cadeia de Markov esparsa fixa (estrutura local).
Cada sequência tem 8 pares (chave, valor) sorteados; o valor NÃO depende da chave (só pode ser lido da definição).
Cada par aparece uma vez como definição `DEF k v` e duas vezes como consulta `QRY k v`, em posições aleatórias.
Avaliação: os valores das DUAS consultas ficam mascarados (a única fonte é a definição) + 15% do preenchimento.
Distância d = |shard(consulta) − shard(definição)|. Acaso = 1/64 ≈ 1,6%.

## Modelos (treinados uma vez, sem escolher checkpoint pelo teste)
- **A (tokens):** Transformer bidirecional d=192, 4 camadas, 6 cabeças, FF 768, posição absoluta. Treino misto 50%
  sequência inteira / 50% janelas (shard + halo 16). Serve aos decodificadores `global`, `isolated`, `halo16`.
- **B (mensagens latentes, a variante "SeedPlane v2"):** mesmo Transformer por shard, SEM halo de tokens; cada shard
  recebe 2 vetores dos vizinhos (esquerda/direita) e emite 2 vetores por rodada. R rodadas; na rodada r cada shard só vê
  as mensagens da rodada r−1 dos vizinhos imediatos. Treinado com R=8 (≥ distância máxima 7).
Gate de sanidade (antes dos critérios): o A global precisa acertar ≥ 90% das consultas com d=0; se falhar, reportar
falha de treino.

## Decodificação e métricas
A: iterativo K=8 com a mesma regra de commit da V6 (cota por shard, argmax). B: R ∈ {1, 2, 4, 8} rodadas, 1 passo.
Acurácia das consultas por distância d ∈ {0, 1, 2–3, 4–7}. 3 seeds de avaliação × 256 sequências. IC95% por bootstrap
sobre sequências. Cobertura (n de consultas) em cada célula.

## Critérios (fixados agora)
- **C0 (tarefa exercida):** A global com d≥2 ≥ 90% E A isolated com d≥2 ≤ 11,6% (acaso + 10 pp), nas 3 seeds.
  Se falhar: INCONCLUSIVO.
- **H1 (halo de tokens, o desenho atual do SeedPlane):** recuperação R_tok = (halo16 − isolated)/(global − isolated)
  para d≥2 ≥ 0,5 nas 3 seeds. **Previsão registrada: falha**, porque o halo só transmite o que já estiver escrito
  como token perto da borda.
- **H2 (mensagens latentes entre vizinhos):** B com R=8 para d≥2 ≥ 0,9 × A global (d≥2), nas 3 seeds.
- **H2-mecanismo (checagem de vazamento):** B com R=2, consultas com d ≥ 4 ≤ 11,6%. A informação não pode andar mais
  rápido que 1 shard por rodada; se andar, há vazamento e H2 é inválido.

## Leitura pré-registrada
H1 falha e H2 passa → a coordenação só entre vizinhos funciona se trocar **estado latente**, não tokens, pagando
latência de ≥ d rodadas para distância d. A ideia é da família de recurrent memory / message passing (não é nova),
mas seria a primeira evidência positiva do SeedPlane. H2 falha → a ideia de só-vizinhos morre nesta forma.
Escopo: tarefa sintética; não mede qualidade de linguagem natural nem velocidade.

## Adendo 1 — treino A falhou no gate de sanidade; curriculum de comprimento (antes de qualquer critério)
Modelo A, 6.000 passos: CE do preenchimento 0,358 (aprendeu a cadeia local), CE das consultas 4,156 = acaso (4,16),
acurácia com d=0 = 5,1% (diagnóstico em 32 sequências da seed 999, fora das seeds de avaliação). O modelo não aprendeu
a recuperação (platô típico de associative recall). Nenhum critério foi calculado.
Correção: curriculum de comprimento em ambos os modelos: passos 0–1.499 com sequências de 128 tokens (1 shard),
1.500–2.999 com 256 (2 shards), 3.000–4.499 com 512 (4 shards), depois 1.024. Pares por sequência = max(2, 8·Lc/1024);
posições absolutas com deslocamento aleatório (múltiplo de 128) para cobrir toda a tabela de posições. O modelo B usa
R = nº de shards da fase. Total 9.000 passos. Janelas do modelo A só na fase de 1.024. Gate de sanidade inalterado.

## Adendo 2 — treino A com curriculum também falhou; causa isolada: token shift (antes de qualquer critério)
Treino 2 (curriculum de comprimento, batch fixo 8): preenchimento não aprendido (CE 6,73 = acaso), consultas 20% em d=0,
loss subindo no final. Diagnóstico (sequências de 256, batch 32, 2.000–6.000 passos, dados da seed 999 fora da avaliação):
nenhuma combinação de lr {1e-3, 3e-4} × clip {0, 1} aprende o bigrama do preenchimento (CE ≈ 6,7); a busca fica em
~50% = "escolher uma das 2 definições visíveis" (só 2 pares por sequência curta), sem casar chave. O gerador foi
conferido (≤ 4 sucessores por token). Posições senoidais não ajudam (consultas 1%).
Causa: platô de otimização — a atenção inicial se dilui sobre todas as posições e o sinal do vizinho some.
Correção (arquitetura, idêntica para A e B e para todos os decodificadores): **token shift** — a entrada soma
W_l·emb(x_{i−1}) + W_r·emb(x_{i+1}), só com vizinhos dentro da janela/shard visível (zero na borda; não vaza
informação entre shards). Teste: consultas 100%, CE 0,012, preenchimento 0,70 em 2.000 passos.
Também: mínimo de 4 pares por sequência curta; batch = 8·1024/Lc (tokens por passo constantes no curriculum);
clip de gradiente 1,0 (estabilidade). Critérios e gate de sanidade inalterados.
