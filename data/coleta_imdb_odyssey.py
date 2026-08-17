"""
================================================================================
 Coletor de User Reviews do IMDb — "The Odyssey" (Christopher Nolan, 2026)
 Base para Sentiment Analysis — SEM API key
================================================================================

O QUE ESTE SCRIPT GERA
-----------------------
Um CSV com as reviews cruas e seus metadados. Ele NÃO gera nenhuma coluna
de sentimento: a classificação é justamente o que você vai produzir depois
(por exemplo, com o léxico ANEW). O rating de 1 a 10 que cada autor deu é
mantido como dado bruto, para servir de gabarito na hora de AVALIAR a sua
classificação — e não como rótulo pronto dentro da base.

POR QUE GRAPHQL E NÃO SCRAPING DE HTML
---------------------------------------
Scraping da página https://www.imdb.com/title/tt33764258/reviews/ com
requests + BeautifulSoup NÃO funciona: o IMDb responde `HTTP 202` com corpo
VAZIO para clientes que não são browser real. O parser nunca recebe HTML,
então o resultado é sempre 0 reviews (testado e confirmado).

Além disso, o esquema antigo de paginação (`/_ajax?paginationKey=...` e o
`<div class="load-more-data">`) é do layout que o IMDb aposentou em 2022 —
esses seletores não existem mais.

A solução é o endpoint GraphQL PÚBLICO que o próprio site do IMDb consome:

    https://caching.graphql.imdb.com/

Não exige API key nem cadastro, devolve JSON estruturado (nada de parsear
HTML que quebra a cada redesign) e entrega todos os campos de uma vez.

Duas pegadinhas descobertas testando o endpoint:
  - os headers `Origin` e `Referer` são obrigatórios (sem eles: HTTP 403)
  - o parâmetro `first` tem teto de 50 por página (100 retorna erro)

COLUNAS DO CSV
--------------
  id             identificador da review no IMDb
  titulo_review  título que o autor deu à review
  texto          corpo da review (é o que você vai analisar)
  rating         nota de 1 a 10 dada pelo autor (vazio em ~2% dos casos)
  autor          nickname do autor
  autor_id       id do autor (útil pra checar duplicidade de pessoa)
  data           data de submissão (AAAA-MM-DD)
  spoiler        se a review foi marcada como spoiler
  votos_util     quantas pessoas marcaram a review como útil
  votos_inutil   quantas marcaram como não útil
  votos_total    soma dos dois
  n_caracteres   tamanho do texto
  n_palavras     contagem de palavras

REQUISITOS
----------
pip install httpx pandas
"""

import html
import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================

MOVIE_ID = "tt33764258"          # The Odyssey (2026), Christopher Nolan
ENDPOINT = "https://caching.graphql.imdb.com/"

PAGE_SIZE = 50                   # 50 é o máximo aceito pelo endpoint (100 dá erro)
PAUSA = 0.6                      # segundos entre requisições — seja gentil
MAX_TENTATIVAS = 4               # retry com backoff em caso de 403/429/5xx
TAMANHO_MINIMO_TEXTO = 20        # descarta reviews vazias/lixo

OUTPUT_CSV = "imdb_odyssey_reviews.csv"
CHECKPOINT = "imdb_odyssey_checkpoint.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    # Origin/Referer são necessários: sem eles o WAF do IMDb devolve 403.
    "Origin": "https://www.imdb.com",
    "Referer": "https://www.imdb.com/",
}

# Ordenar por data de submissão deixa a coleta determinística (a ordem
# "mais úteis primeiro" muda a cada minuto conforme os votos entram, o que
# provoca reviews duplicadas e outras puladas durante a paginação).
QUERY = """
query Reviews($id: ID!, $first: Int!, $after: ID) {
  title(id: $id) {
    titleText { text }
    releaseYear { year }
    ratingsSummary { aggregateRating voteCount }
    reviews(first: $first, after: $after,
            sort: { by: SUBMISSION_DATE, order: DESC }) {
      total
      pageInfo { hasNextPage endCursor }
      edges {
        node {
          id
          author { nickName userId }
          summary { originalText }
          text { originalText { plainText } }
          authorRating
          submissionDate
          spoiler
          helpfulness { upVotes downVotes }
        }
      }
    }
  }
}
"""


# ==============================================================================
# 2. CAMADA DE REDE
# ==============================================================================

def consultar(client: httpx.Client, after: str | None) -> dict:
    """Faz uma chamada GraphQL com retry e backoff exponencial."""
    payload = {
        "query": QUERY,
        "variables": {"id": MOVIE_ID, "first": PAGE_SIZE, "after": after},
    }

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resp = client.post(ENDPOINT, json=payload)
        except Exception as e:
            espera = 2 ** tentativa
            print(f"    (erro de rede: {type(e).__name__} — retry em {espera}s)")
            time.sleep(espera)
            continue

        if resp.status_code == 200:
            dados = resp.json()
            if dados.get("errors"):
                raise RuntimeError(f"GraphQL retornou erro: {dados['errors']}")
            return dados["data"]["title"]

        # 403 = WAF (rajada rápida demais), 429 = rate limit, 5xx = instabilidade
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            espera = 2 ** tentativa
            print(f"    (HTTP {resp.status_code} — aguardando {espera}s e tentando de novo)")
            time.sleep(espera)
            continue

        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    raise RuntimeError(f"Falhou após {MAX_TENTATIVAS} tentativas (cursor={after})")


# ==============================================================================
# 3. LIMPEZA E NORMALIZAÇÃO
# ==============================================================================

def limpar_texto(texto: str | None) -> str:
    """Normaliza espaços, desfaz entidades HTML e remove tags residuais."""
    if not texto:
        return ""
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", " ", texto)       # <br> etc. que às vezes escapam
    texto = texto.replace("​", "")          # zero-width space
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def achatar(node: dict) -> dict:
    """Converte um nó do GraphQL numa linha plana do dataset."""
    texto = limpar_texto(
        ((node.get("text") or {}).get("originalText") or {}).get("plainText")
    )
    titulo = limpar_texto((node.get("summary") or {}).get("originalText"))
    ajuda = node.get("helpfulness") or {}
    autor = node.get("author") or {}
    up, down = ajuda.get("upVotes", 0), ajuda.get("downVotes", 0)

    return {
        "id": node.get("id"),
        "titulo_review": titulo,
        "texto": texto,
        "rating": node.get("authorRating"),
        "autor": autor.get("nickName"),
        "autor_id": autor.get("userId"),
        "data": node.get("submissionDate"),
        "spoiler": bool(node.get("spoiler")),
        "votos_util": up,
        "votos_inutil": down,
        "votos_total": up + down,
        "n_caracteres": len(texto),
        "n_palavras": len(texto.split()),
    }


# ==============================================================================
# 4. COLETA
# ==============================================================================

def coletar() -> tuple[list[dict], dict]:
    """Pagina por todas as reviews. Salva checkpoint pra poder retomar."""
    registros: list[dict] = []
    vistos: set[str] = set()
    after: str | None = None
    pagina = 0
    info_filme: dict = {}

    # Retoma de um checkpoint anterior, se existir
    if Path(CHECKPOINT).exists():
        salvo = json.loads(Path(CHECKPOINT).read_text(encoding="utf-8"))
        registros = salvo["registros"]
        vistos = {r["id"] for r in registros}
        after = salvo.get("after")
        pagina = salvo.get("pagina", 0)
        info_filme = salvo.get("info_filme", {})
        print(f"Retomando checkpoint: {len(registros)} reviews ja coletadas.\n")

    with httpx.Client(headers=HEADERS, timeout=40, follow_redirects=True) as client:
        while True:
            titulo_dados = consultar(client, after)

            if not info_filme:
                resumo = titulo_dados.get("ratingsSummary") or {}
                info_filme = {
                    "titulo": (titulo_dados.get("titleText") or {}).get("text"),
                    "ano": (titulo_dados.get("releaseYear") or {}).get("year"),
                    "nota_media_imdb": resumo.get("aggregateRating"),
                    "total_votos_imdb": resumo.get("voteCount"),
                }
                print(f"Filme: {info_filme['titulo']} ({info_filme['ano']}) - "
                      f"nota IMDb {info_filme['nota_media_imdb']} "
                      f"({info_filme['total_votos_imdb']:,} votos)\n")

            reviews = titulo_dados["reviews"]
            total = reviews["total"]
            novos = 0

            for edge in reviews["edges"]:
                linha = achatar(edge["node"])
                if linha["id"] in vistos:
                    continue
                if linha["n_caracteres"] < TAMANHO_MINIMO_TEXTO:
                    continue
                vistos.add(linha["id"])
                registros.append(linha)
                novos += 1

            pagina += 1
            print(f"  pagina {pagina:>3}: +{novos:>3} novas "
                  f"(acumulado {len(registros):>5} de ~{total})")

            if not reviews["pageInfo"]["hasNextPage"]:
                print("\n  Fim da paginacao.")
                break

            after = reviews["pageInfo"]["endCursor"]

            # checkpoint a cada 10 páginas
            if pagina % 10 == 0:
                Path(CHECKPOINT).write_text(json.dumps(
                    {"registros": registros, "after": after,
                     "pagina": pagina, "info_filme": info_filme},
                    ensure_ascii=False), encoding="utf-8")

            time.sleep(PAUSA)

    return registros, info_filme


# ==============================================================================
# 5. RELATÓRIO DE QUALIDADE
# ==============================================================================

def relatorio(df: pd.DataFrame, info: dict) -> None:
    print("\n" + "=" * 70)
    print(" RELATORIO DE QUALIDADE DA BASE")
    print("=" * 70)

    print(f"\nFilme: {info.get('titulo')} ({info.get('ano')})")
    print(f"Nota media no IMDb: {info.get('nota_media_imdb')} "
          f"({info.get('total_votos_imdb'):,} votos)")

    print(f"\nTotal de reviews coletadas:  {len(df):,}")
    print(f"Com rating numerico:         {df['rating'].notna().sum():,} "
          f"({df['rating'].notna().mean():.1%})")
    print(f"Autores unicos:              {df['autor_id'].nunique():,}")
    print(f"Marcadas como spoiler:       {df['spoiler'].sum():,}")
    print(f"Periodo coberto:             {df['data'].min()} a {df['data'].max()}")

    print("\nTamanho do texto (caracteres):")
    print(f"  min={df.n_caracteres.min():,}  mediana={int(df.n_caracteres.median()):,}  "
          f"media={int(df.n_caracteres.mean()):,}  max={df.n_caracteres.max():,}")

    print("\nDistribuicao dos ratings (1-10):")
    dist = df["rating"].value_counts().sort_index()
    for nota, qtd in dist.items():
        barra = "#" * max(1, int(60 * qtd / dist.max()))
        print(f"  {int(nota):>2} | {qtd:>5} {barra}")

    # Sanidade: a média dos ratings tem que ficar na mesma vizinhança da
    # nota pública do IMDb. Se destoasse muito, seria sinal de coleta enviesada.
    media = df["rating"].mean()
    oficial = info.get("nota_media_imdb")
    print(f"\nSanidade: media dos ratings coletados = {media:.2f} | "
          f"nota publica do IMDb = {oficial}")
    if oficial:
        print(f"  diferenca = {abs(media - oficial):.2f} (diferencas pequenas sao "
              f"esperadas: quem escreve review nao e a mesma populacao de quem so vota)")

    print("\nObs.: a base NAO contem coluna de sentimento - a classificacao e o que")
    print("      voce vai gerar. O 'rating' fica disponivel apenas como gabarito")
    print("      para avaliar essa classificacao depois.")


# ==============================================================================
# 6. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print(f"Coletando reviews do IMDb para {MOVIE_ID} via GraphQL publico...\n")
    registros, info = coletar()

    if not registros:
        print("\nNenhuma review coletada. Verifique a conexao e o MOVIE_ID.")
        return

    df = pd.DataFrame(registros)

    # Dedup: por id (garantido pela coleta) e por texto idêntico (mesma pessoa
    # postando duas vezes, ou repost) — mantém a review com mais votos.
    antes = len(df)
    df = (df.sort_values("votos_total", ascending=False)
            .drop_duplicates(subset=["texto"], keep="first")
            .reset_index(drop=True))
    if antes != len(df):
        print(f"\nDeduplicacao por texto: {antes - len(df)} review(s) repetida(s) removida(s).")

    df = df.sort_values("data", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    relatorio(df, info)

    print(f"\nArquivo salvo: {OUTPUT_CSV} ({len(df):,} linhas, {len(df.columns)} colunas)")
    print("Colunas:", ", ".join(df.columns))

    if Path(CHECKPOINT).exists():
        Path(CHECKPOINT).unlink()


if __name__ == "__main__":
    main()
