# SeedPlane V5 — protocolo antes dos resultados

Objetivo: testar se o roteamento Hadamard oferece vantagem sobre identidade explícita e corrigir falhas de integridade sem reinterpretar a correção como descoberta. Preservar V4.

Seeds 11,23,37. Cenários: limpo, fronteira errada, colisão modulo32, mesma fronteira/geração antiga, requisição errada, versão de modelo errada, alvo errado e duplicação. Gold apenas decide validade no evaluator. Comparar sem filtro, boundary ID, Hadamard32, envelope exato+deduplicação e V5 (envelope exato+Hadamard+deduplicação).

Medir falsas aceitações/rejeições, latência por mensagem, bytes de metadados; workers persistentes reais com processos separados e entrega fora de ordem. IDs/versionamento são baseline obrigatório. Benefício assíncrono compartilhado por todos os roteadores não é ganho da SeedPlane.

Verificar checkpoint toy em inferência real por shards com workers persistentes, 1/2/4 workers, latência, RSS e NLL mascarada. Separar startup de execução, tamanho do payload e tempo do roteador. Três seeds de agendamento/dados não são três treinamentos. Protocolar corpus/checkpoint: não retreinar nem selecionar pesos pelo teste.

Critério para vantagem nesta rodada: zero mensagens inválidas aceitas nos cenários e qualidade equivalente, além de melhoria de pelo menos 10% em tempo total em comparação pareada com o envelope exato, consistente nas três seeds. Ganho só no microbenchmark não basta. ICs exploratórios por repetição; nenhuma alegação de escala grande ou revolução a partir do modelo toy.

Correção prevista: V5 inclui request, generation, boundary, model_version, target e source na validação, e deduplicação. Ambas as alternativas usam o mesmo contrato de validade. Comparar metadados como transmitidos; não dar IDs grátis a um roteador e cobrar do outro. Dados e latências sintéticas serão claramente identificados.
