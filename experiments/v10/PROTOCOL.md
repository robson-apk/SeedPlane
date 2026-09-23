# SeedPlane V10 — dispositivos heterogêneos: GPU + CPU + Mac pela rede (protocolo antes dos resultados, 2026-09-23)

## Pergunta
O SeedPlane escala somando dispositivos diferentes (Intel Arc B580, núcleos do Ryzen 5600X, núcleos do Mac M4 pela LAN),
sincronizando os 16 passos de refinamento? E como se compara ao Transformer tradicional nos mesmos dispositivos?

## Endpoints (todos servidores `multiprocessing.connection`; o coordenador roda no 5600X)
- `gpu`: 1 processo no 5600X usando a B580 (XPU, fp32).
- `cpu4`: 4 processos no 5600X, 1 thread cada (2 núcleos ficam livres para o coordenador e o driver da GPU).
- `mac2`: 2 processos no Mac, 1 thread cada, prioridade baixa (`nice`), acessados pela LAN (10.0.0.92).
Mesmo checkpoint (`checkpoints/v6_mdlm_d256_l6_seed1.pt`) em todos.

## Métodos
- **SeedPlane:** as unidades são (página, shard); o coordenador distribui as unidades entre os endpoints na proporção
  da velocidade medida numa calibração antes de cada configuração; a cada passo envia o estado atual e recebe
  (token, confiança) só das posições mascaradas; regra de commit da V8/V9.
- **Tradicional:** latência = atenção global no melhor dispositivo sozinho (e na CPU, como referência);
  vazão = paralelismo de dados, páginas inteiras distribuídas entre os mesmos endpoints na proporção da velocidade.

## Conjuntos de dispositivos
{gpu} · {cpu4} · {gpu, cpu4} · {cpu4, mac2} · {gpu, cpu4, mac2}.

## Modos
- **Latência:** 1 página de 1.024 tokens por vez; 3 seeds × 8 páginas.
- **Vazão:** lote de 32 páginas; 3 seeds × 1 lote.
50% mascarado; 16 passos. Métricas: tempo, tokens/s (tokens escondidos preenchidos por segundo), tempo de espera por
sincronização (parede − maior tempo de cálculo reportado pelos endpoints), concordância de tokens com {gpu} sozinho.

## Critérios (fixados agora)
- **H1 (latência escala com dispositivos):** SeedPlane no melhor conjunto multi-dispositivo ≤ 0,90 × SeedPlane em {gpu},
  nas 3 seeds (IC95% superior da razão < 0,90).
- **H2 (latência vs tradicional):** SeedPlane no melhor conjunto < Tradicional em {gpu} (IC superior < 1), 3 seeds.
- **H3 (vazão escala):** SeedPlane {gpu, cpu4, mac2} ≥ 1,10 × SeedPlane {gpu} em tokens/s, 3 seeds.
- **H4 (vazão vs tradicional com os mesmos dispositivos):** SeedPlane ≥ Tradicional em tokens/s no mesmo conjunto
  {gpu, cpu4, mac2}, 3 seeds.
- **Q:** concordância de tokens de cada conjunto com {gpu} ≥ 99% (diferenças de ponto flutuante entre dispositivos).
Previsões registradas: a B580 domina; em latência, somar CPU e Mac provavelmente NÃO ajuda (sincronização a cada passo,
rede para o Mac) → H1 provavelmente falha. Em vazão, somar dispositivos tende a ajudar os dois métodos.
Escopo: modelo pequeno (5,3M); o resultado de dispositivos lentos somados a uma GPU pode mudar com modelos grandes.
