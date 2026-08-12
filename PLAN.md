Assumindo que seja *The Odyssey* (2026), de Christopher Nolan, eu começaria com um piloto de 300–500 comentários em inglês publicados após a estreia, em 17 de julho de 2026. O objetivo não seria apenas classificar “positivo/negativo”, mas medir as três dimensões do ANEW: valência, ativação e dominância. A estreia nessa data consta no material oficial da Universal. [Trailer oficial](https://www.youtube.com/watch?v=Mzw2ttJD2qQ)

## 1. Definam a pergunta de pesquisa

Uma formulação boa e viável:

> Como varia a resposta emocional ao filme *The Odyssey* nas comunidades do Reddit durante o primeiro mês após sua estreia?

Perguntas secundárias:

- A recepção foi predominantemente positiva ou negativa?
- Comentários negativos possuem maior ativação emocional?
- Subreddits de fãs apresentam maior valência que comunidades gerais de cinema?
- A recepção mudou entre a semana de estreia e as semanas seguintes?

Importante: no ANEW, **valência** representa agradável/desagradável; **arousal** representa intensidade emocional; **dominance** representa sensação de controle ou poder. Arousal alto não significa necessariamente sentimento negativo.

## 2. Delimitem a amostra

Sugestão inicial:

- Período: **17/07/2026 a 12/08/2026**.
- Idioma: inglês.
- Unidade de análise: comentário de primeiro nível, não cada frase.
- Tamanho mínimo: 300–500 comentários para o piloto; 2.000 ou mais para o estudo final.
- Comprimento mínimo: 15 ou 20 palavras.
- Comunidades gerais: `r/movies`, `r/moviecritic`, `r/moviereviews`, `r/TrueFilm`.
- Comunidades de fãs: `r/ChristopherNolan`, `r/TheOdysseyMovie` ou equivalentes.

Já existe, por exemplo, um [tópico de reviews em r/movies](https://www.reddit.com/r/movies/comments/1ux9xiv/christopher_nolans_the_odyssey_review_thread/). Não usem o corpo desse post como opinião de usuários, pois ele contém trechos de críticos profissionais; usem somente os comentários relevantes.

Filtrem pesquisas com combinações como:

```text
"The Odyssey" Nolan
"The Odyssey" review
"Odyssey 2026"
"Odyssey" IMAX
```

Para garantir que sejam opiniões de quem assistiu, priorizem tópicos pós-estreia e comentários contendo expressões como `I saw`, `I watched`, `just got out`, `the movie was` ou `my review`.

## 3. Coletem os campos necessários

Uma tabela simples poderia ter:

```text
comment_id
submission_id
parent_id
subreddit
created_utc
body
score
depth
permalink
collected_at
```

Não tratem o número de upvotes como sentimento. Guardem-no apenas como variável de engajamento.

Evitem armazenar nomes de usuário. Também será necessário excluir registros cujo conteúdo tenha sido apagado posteriormente. Atualmente, o Reddit exige OAuth, identificação adequada do cliente e cumprimento dos limites da API; a documentação informa limite gratuito de até 100 consultas por minuto por cliente aprovado. [Documentação da Data API](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)

Se for um projeto universitário, o melhor caminho é o [Reddit for Researchers](https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program). O programa é gratuito para pesquisas acadêmicas não comerciais aprovadas e fornece os dados pelo BigQuery. Não recomendo scraping direto das páginas.

## 4. Escolham corretamente o léxico

Para comentários em inglês, o ANEW original tem somente 1.034 palavras, o que provavelmente resultará em baixa cobertura. Uma escolha melhor é a extensão de Warriner, Kuperman e Brysbaert, com **13.915 lemas ingleses** e valores de valência, arousal e dominance. [Artigo e descrição do dataset](https://macsphere.mcmaster.ca/handle/11375/22965)

Se a disciplina exigir especificamente o ANEW original:

- executem o experimento com o ANEW original;
- repitam com a extensão de Warriner;
- comparem a cobertura e os resultados.

Não traduzam automaticamente os comentários para português antes da análise. Caso coletem textos em português, usem o ANEW-Br, que contém 1.046 palavras e medidas de valência e ativação, mas não a mesma cobertura de dominância. [ANEW-Br](https://pubmed.ncbi.nlm.nih.gov/25924086/)

## 5. Pipeline de processamento

Para cada comentário:

1. Remover URLs, citações em Markdown, marcações de spoiler e conteúdo `[deleted]`.
2. Detectar o idioma e manter apenas inglês.
3. Converter para minúsculas.
4. Tokenizar e lematizar: `loved → love`, `disappointing → disappoint`.
5. Expandir contrações: `wasn't → was not`.
6. Cruzar os lemas com o léxico.
7. Calcular as médias:

```text
valence_comentario  = média da valência das palavras encontradas
arousal_comentario  = média da ativação das palavras encontradas
dominance_comentario = média da dominância das palavras encontradas

coverage = palavras encontradas no ANEW / palavras relevantes do comentário
```

Mantenham a valência como variável contínua. Se precisarem de classes e a escala for de 1 a 9, uma regra inicial possível é:

```text
valência < 4,5  → negativa
4,5–5,5         → neutra
valência > 5,5  → positiva
```

Esses limites precisam ser apresentados como decisão metodológica, não como regra oficial do ANEW.

Implementem pelo menos uma correção simples de negação. Em uma escala de 1 a 9, para uma palavra no alcance de `not`, `never` ou `no`, pode-se usar:

```text
valência_corrigida = 10 - valência_original
```

Por exemplo, `good = 7,5` passa aproximadamente para `2,5` em `not good`.

## 6. Validem antes de confiar nos resultados

Selecionem aleatoriamente cerca de 150–200 comentários, equilibrados por subreddit, e peçam para duas pessoas classificarem manualmente como positivo, neutro ou negativo.

Depois comparem:

- concordância entre os avaliadores, usando Cohen’s Kappa;
- classes manuais versus valência do ANEW;
- acurácia e macro-F1;
- cobertura lexical média;
- principais erros: sarcasmo, negação, gírias e comentários sobre cenas violentas.

Uma cena descrita com palavras negativas pode ser elogiada pelo usuário — por exemplo, “the terrifying scene was incredible”. Esse tipo de contraste é uma limitação importante de métodos lexicais.

O primeiro entregável do grupo deveria ser um protocolo de uma página contendo pergunta, período, subreddits, critérios de inclusão/exclusão, campos coletados e léxico escolhido. Depois disso, vocês já podem construir e validar o piloto antes de ampliar a coleta.