# SeedPlane V6 — resultado (2026-09-23)

**Veredito pré-registrado: INCONCLUSIVO.** O critério C0 falhou: o modelo treinado quase não usa contexto de longo
alcance. Por isso a pergunta central da V6 ("a troca iterativa só entre vizinhos recupera informação distante?")
**não foi testada**. H1 e H2 não são interpretados.

## Caminho até o modelo (registrado em PROTOCOL.md, adendos 1–4)
- Treinos 1–3: colapsaram no platô do unigrama e falharam no gate de sanidade (loss com 15% de máscara 4,92–4,96;
  unigrama 5,10). A atribuição ao gradient clipping (adendo 3) **foi retirada**: sair do platô é estocástico
  (adendo 4).
- Treino 4 (curriculum de máscara, seed 1): passou no gate com loss 1,70 com 15% de máscara.

## C0 — o modelo usa contexto distante? (L=1024, máscara 0,5, K=16; subconjunto LR ≈ 3.200 tokens por seed)
| Seed | global − isolated (LR) | IC95% |
|---|---|---|
| 11 | +0,64 pp | [−0,34; 1,60] |
| 23 | +1,65 pp | [0,83; 2,45] |
| 37 | −0,03 pp | [−0,80; 0,83] |
Exigido: ≥ 2 pp com IC acima de 0 nas três seeds → **falhou**.

## Observação lateral (não é critério)
O `halo16` teve 0,59–0,67 pp de acurácia **maior** que o decodificador global em todos os tokens, nas três seeds.
Leitura provável (inferida): metade do treino usou janelas locais, e o TinyStories tem histórias curtas (~200 palavras).
Uma janela de 1.024 tokens atravessa várias histórias, e o contexto distante é quase sempre de outra história.

## O que seria preciso para testar a pergunta
Um corpus com dependências longas reais: documentos longos, ou uma tarefa sintética de cópia a distância. O C0
precisa passar **antes** de H1/H2 terem significado. Isso exige um protocolo novo.

Dados: `results/eval_rows.json` (por sequência), `results/analysis.json`, logs de treino `results/run*_*`.
Checkpoint: `checkpoints/v6_mdlm_d256_l6_seed1.pt` (21 MB, 5,26M parâmetros, seed 1, 10k passos).
