# SeedPlane V5 — auditoria executada

**Resultado: não foi demonstrada vantagem incremental das chaves Hadamard sobre IDs + versão.** Correções de integridade funcionam; o baseline simples obtém as mesmas saídas. Nenhuma descoberta revolucionária foi estabelecida.

## O que foi feito

Preservamos V4, seus pesos e resultados. Implementamos protocolo com critérios prévios, quatro scripts de avaliação, envelope versionado com deduplicação, runtime real com workers persistentes e três testes de engenharia. Seeds 11/23/37 são de amostragem/agendamento, não três treinamentos.

- 54.000 mensagens distintas de teste, avaliadas pelos cinco roteadores (270.000 decisões).
- 108 batches de inferência real com 1/2/4 workers, comprimentos 512/1024, sem atraso e com jitter controlado.
- 180 comparações pareadas independentes (360 chamadas de inferência+IPC+fusão) entre envelope exato e V5.
- 30 cenários de gerações concorrentes, com 120 inferências antigas e 120 atuais realmente calculadas.
- Três testes de engenharia passaram, incluindo equivalência modulo32 e validação de todos os campos.

## Correção e falsificações

Nos testes de metadados, cada condição tem 6.000 exemplos nas três seeds. V4 rejeita fronteiras diferentes sem colisão, mas aceita 100% dos casos injetados de colisão +32, geração antiga, requisição errada, versão errada, alvo errado, origem errada e duplicação. IDs apenas resolvem colisões e fronteira, mas não versão. Envelope exato e V5 não cometeram erros nos casos testados.

A V5 valida `(request, generation, boundary, model_version, target, source)` e deduplica antes de aceitar. `source` é o shard lógico, não o processo físico: migração entre workers não muda a identidade. A validade é relativa ao envelope esperado; versão e deduplicação não detectam corrupção do payload que preserve metadados. Para isso seriam necessários controles de integridade adicionais igualmente aplicáveis aos dois métodos.

### Mensagens antigas de verdade

| Roteador | Antigas aceitas / 120 | Atuais aceitas / 120 |
|---|---:|---:|
| unfiltered | 120 | 120 |
| boundary_id | 120 | 120 |
| hadamard_v4 | 120 | 120 |
| exact_envelope | 0 | 120 |
| seedplane_v5 | 0 | 120 |

As entradas antigas e atuais foram diferentes, e ambas foram inferidas pelo checkpoint. O atraso de 25 ms foi imposto no experimento; não é latência de uma rede medida.

## Qualidade

Nos cenários de corrupção controlada, a V4 deixa passar colisões, requisições erradas e estados antigos; o NLL médio de fronteira piorou aproximadamente 0,0966 nesses cenários. Esse número pertence à perturbação com vocabulário permutado, não ao teste de mensagens antigas reais nem à qualidade natural de geração.

Envelope exato e V5 conservaram o resultado limpo. Nas 180 comparações pareadas, a diferença máxima entre as probabilidades finais dos dois foi **zero**. Isso não prova qualidade linguística elevada: ambos usam o mesmo checkpoint toy.

## Tempo completo: teste decisivo

Ordem dos roteadores randomizada dentro de cada par, mesmos inputs/jitter, processos persistentes. IC95% por bootstrap de 2.000 reamostragens de pares; exploratório, sem correção por multiplicidade. Redução positiva favorece V5.

| Workers | Seed | IDs+versão mediana ms | V5 mediana ms | Redução média % | IC95% % |
|---:|---:|---:|---:|---:|---|
| 1 | 11 | 41.895 | 40.749 | -0.26 | [-2.64; 2.55] |
| 1 | 23 | 43.429 | 45.270 | 0.25 | [-3.02; 3.18] |
| 1 | 37 | 41.534 | 41.910 | -1.28 | [-4.98; 2.12] |
| 2 | 11 | 20.602 | 20.535 | 0.61 | [-0.87; 2.01] |
| 2 | 23 | 22.008 | 21.950 | 0.55 | [-0.41; 1.49] |
| 2 | 37 | 22.684 | 22.447 | -0.80 | [-2.03; 0.45] |
| 4 | 11 | 11.953 | 12.561 | -4.43 | [-8.36; -1.05] |
| 4 | 23 | 12.956 | 12.878 | 0.54 | [-0.78; 1.81] |
| 4 | 37 | 13.283 | 13.536 | -2.34 | [-3.90; -1.03] |

**Nenhuma configuração passou o critério de ganho mínimo de 10%.** No microbenchmark de decisões Python, Hadamard V4 também não superou a comparação de IDs; comparar funções escalarizadas não estabelece limite de desempenho de toda implementação vetorizada possível.

O benchmark de inferência apresentou ganho com mais workers, compartilhado por ambos os roteadores. Por exemplo, agregando seeds e condições de jitter em L=1024, medianas de aproximadamente 25,26 / 13,94 / 7,92 ms com 1/2/4 workers. Não atribuir essa aceleração às chaves. São medições desta máquina, modelo pequeno e payload: cerca de 5,11 MB por batch nesse comprimento. Não extrapolar para LLMs grandes.

## Por que a comparação é decisiva neste mecanismo

Para as linhas usadas da matriz Hadamard, o score de correspondência identifica precisamente `boundary_source % 32 == boundary_target % 32`. Sem colisão, isso reexpressa igualdade de ID. Com envelope validado, o teste Hadamard adicional é redundante. Alterar o tamanho da matriz pode ampliar o domínio sem colisão, mas não cria informação extra sobre request, geração ou payload.

Transmitir uma chave de 32 float32 custaria 128 bytes; regenerá-la proceduralmente do ID evita esse tráfego. Portanto, esses bytes são uma hipótese de formato, não custo inevitável nem medição de rede. O runtime transporta probabilidades via IPC local; nenhum ganho de tráfego foi demonstrado para SeedPlane.

## Escopo e decisão

O runtime implementa inferência por shards reais usando a cabeça central do checkpoint, e fusão owner/halo. Não é o protocolo original de cinco cabeças completo, difusão iterativa convergente nem geração livre distribuída. Corpus/vocabulário foram reconstruídos pelo loader do projeto; a identidade com o tokenizer do treinamento original não tem verificação independente. Essa limitação não altera a igualdade pareada dos roteadores, mas restringe a interpretação do NLL absoluto.

A solução corrigida é útil como engenharia de coordenação. A hipótese de que **as chaves ortogonais, neste desenho, acrescentam uma vantagem frente ao envelope exato foi rejeitada nesta bateria**. Não aumentar repetições ou selecionar seeds favoráveis para renomear ruído como descoberta. Uma contribuição futura exigirá mecanismo e benefício que o envelope simples não forneça.

## Arquivos e reprodução

Na pasta do projeto, com Python e dependências do projeto instaladas:

```sh
OPENBLAS_NUM_THREADS=1 python research_v5/audit_routing.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research_v5/runtime_benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research_v5/paired_runtime.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research_v5/live_stale_test.py
python -m unittest discover -s research_v5 -p 'test_*.py'
python research_v5/summarize.py
```

O modelo e o corpus continuam nos caminhos existentes do projeto. Nenhum arquivo original foi reescrito para obter os resultados. O README principal recebe somente aviso/link da auditoria, preservando o texto histórico. Ver `PROTOCOL.md`, `summary.json` e resultados por exemplo.
