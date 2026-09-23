# CLMP v4 — SeedPlane procedural para coordenação de shards

## Mudança principal

A seed saiu do hidden state. Em vez de ser somada aos embeddings, ela agora funciona como uma camada de coordenação acima dos shards/cores.

Para cada fronteira `b`, a SeedPlane gera uma chave procedural ortogonal `K_b` (código de Hadamard, sem parâmetros treináveis):

- lado owner/core: `+K_b`
- halo emprestado: `-K_b`
- compatibilidade: `max(0, -cos(K_source, K_owner))`

Assim, um halo complementar da fronteira certa tem score ~= 1 e uma mensagem de outra fronteira tem score ~= 0. A SeedPlane pode filtrar/rutear mensagens antes do confidence linking/sparse linking.

## Por que esta posição é mais coerente

A seed não precisa carregar semântica do texto; ela descreve topologia/coordenação entre workers. Colocá-la no hidden state deixa o Transformer livre para ignorá-la. Como metadado de roteamento, ela tem função obrigatória e não perturba o backbone.

## Controle limpo

No corpus/validação sincronizada, a seed por si só não trouxe ganho de qualidade detectável. Seed sinusoidal real, constante e aleatória ficaram praticamente empatadas. O pequeno ganho visto na fusão vinha do método de fusão, não do vetor. Isso é esperado quando todos os chunks já chegam corretamente identificados.

Em 1024, para a fusão PoE testada:
- linker padrão: NLL ~5.12178249
- PoE com seed sinusoidal: NLL ~5.12176967 em uma amostra inicial
- em 20 batches independentes, delta médio total ~+1.3e-6 (ruído), boundary delta médio ~-4.44e-5
- seed constante teve resultado praticamente idêntico

Conclusão: a seed não deve ser vendida como ganho de perplexidade em execução sincronizada.

## Tentativa de usar seed como attention bias

Foi testado um bias de attention que favorecia pares `+S/-S` na mesma fronteira. O mecanismo alterava de fato os logits, mas o `float attention mask` usado pelo TransformerEncoder do PyTorch derrubou o fast-path nesta CPU. O treino ficou muitas vezes mais lento e até 150 passos a curva de loss era indistinguível do CLMP normal. Esta implementação foi rejeitada como caminho de runtime.

## SeedPlane ortogonal

Com `KEY_DIM=32`, usando linhas de uma matriz de Hadamard:

- afinidade complementar correta: 0.99999994
- maior |cos| entre 8 fronteiras diferentes: ~5.96e-8
- custo de matching 8x8: ~0.005 ms
- custo de matching 32x32: ~0.006 ms

Com shard=128, 32 chaves cobrem até 4096 tokens antes de ser necessário ampliar/hierarquizar o espaço de chaves.

## Stress test assíncrono

Foi simulado cross-talk/staleness: propostas de halo vindas de outra fronteira foram misturadas e também tiveram o vocabulário permutado, representando uma mensagem errada/corrompida. Esse é um teste sintético de robustez do roteador, não um benchmark de qualidade natural.

### Contexto 1024 — aumento de boundary NLL versus caminho limpo

| Cross-talk | Sem SeedPlane | Com SeedPlane |
|---:|---:|---:|
| 10% | +0.0002964 | +0.0000000 |
| 25% | +0.0009342 | +0.0000000 |
| 50% | +0.0024969 | +0.0000000 |

### Contexto 512

| Cross-talk | Sem SeedPlane | Com SeedPlane |
|---:|---:|---:|
| 10% | +0.0008233 | +0.0000000 |
| 25% | +0.0015663 | +0.0000000 |
| 50% | +0.0029787 | +0.0000000 |

O zero ocorre porque, neste teste, as chaves de Hadamard das fronteiras usadas são ortogonais: a mensagem estrangeira recebe compatibilidade zero e é descartada integralmente.

## Interpretação

A SeedPlane não é uma melhoria de linguagem. Ela é um protocolo de coordenação para tornar shards autônomos/assíncronos sem perder a identidade das inferências complementares.

Arquitetura atual:

```
                     SeedPlane
       K0             K1             K2
       |              |              |
   +---+---+      +---+---+      +---+---+
   | core 0| halo | core 1| halo | core 2|
   +---+---+      +---+---+      +---+---+
        \ -K0   +K0 / \ -K1   +K1 /
             complementary routing
                      |
              sparse top-K linker
                      |
              confidence fusion
```

A combinação recomendada para o próximo runtime é:

`CLMP backbone sharded + sparse top-K=16 + SeedPlane boundary keys + confidence fusion + workers persistentes assíncronos`.

## Limitações

- Modelo toy (~84.7k parâmetros), não LLM grande.
- CPU only.
- Stress test de cross-talk é deliberadamente sintético.
- O benefício da SeedPlane deve aparecer principalmente quando shards forem assíncronos, reordenados, migrados ou houver propostas stale; em execução perfeitamente sincronizada, a topologia já fornece essa informação gratuitamente.
