# -*- coding: utf-8 -*-
"""
Classificacao de sentimento das reviews de "The Odyssey" (2026) com o ANEW.

ENTRADAS
    data/anew.csv                   lexico ANEW (Bradley & Lang, 1999)
    data/imdb_odyssey_reviews.csv   corpus cru, sem nenhum rotulo de sentimento

SAIDA (uma unica)
    imdb_odyssey_reviews_anew.csv   o corpus original + a coluna 'sentimento'
                                    ("positivo" ou "negativo" para toda review)

O MODELO, EM CINCO PASSOS
    1. cada palavra do texto e procurada no ANEW;
    2. 'pleasure' define a POLARIDADE da palavra (agradavel <-> desagradavel);
    3. 'arousal' define apenas a INTENSIDADE - nunca troca o sinal;
    4. o score da review e a media das polaridades ponderada pela intensidade,
       normalizada para o intervalo [-1, +1];
    5. score >= LIMIAR_DECISAO -> "positivo"; caso contrario -> "negativo".

A ESCALA DO ANEW (detalhe que muda o resultado)
    O manual de Bradley & Lang usa o SAM, uma escala de 9 pontos por dimensao,
    em que 1 = completamente infeliz, 9 = completamente feliz e *5 = neutro*
    ("neither happy nor sad", nas instrucoes dadas aos participantes).
    O data/anew.csv nao esta nessa escala: seus valores sao os do artigo
    multiplicados por 100/8.82, onde 8.82 e o maior 'pleasure' do ANEW
    (a palavra "triumphant", que assim vira 100.0). Conferido termo a termo
    contra a Tabela 1 do artigo: a razao csv/artigo e 11.337868 para as tres
    dimensoes.
    Consequencia pratica: o ponto neutro no CSV e 5 * 100/8.82 = 56.69, e nao
    50. Tratar 50 como neutro contaria como "agradavel" toda palavra entre
    50 e 56.69, que o artigo classifica como desagradavel. Por isso aqui os
    valores voltam para a escala 1-9 antes de qualquer conta.

POR QUE 'dominance' NAO ENTRA
    O artigo descreve dominance como o eixo controlado <-> no controle, o mais
    fraco dos tres e sem relacao com agradavel/desagradavel. Ele nao informa
    polaridade, entao fica de fora do score (a coluna continua no CSV lido).

Sem dependencias externas: apenas a biblioteca padrao. Rode da raiz do repo:
    python anew_classificador.py
"""

from __future__ import annotations

import csv
import re
import sys

# --------------------------------------------------------------- arquivos ----
ANEW_CSV = "data/anew.csv"
REVIEWS_CSV = "data/imdb_odyssey_reviews.csv"
SAIDA_CSV = "imdb_odyssey_reviews_anew.csv"

COL_TERMO, COL_PLEASURE, COL_AROUSAL = "term", "pleasure", "arousal"

# ----------------------------------------------------------------- escala ----
# Desfaz o reescalonamento do CSV e devolve os valores a escala SAM 1-9 do
# artigo. 8.82 = maior 'pleasure' do ANEW; ver secao "A ESCALA DO ANEW".
MAX_PLEASURE_ARTIGO = 8.82
FATOR_PARA_SAM = MAX_PLEASURE_ARTIGO / 100.0

SAM_NEUTRO = 5.0        # ponto neutro declarado nas instrucoes do SAM
SAM_AMPLITUDE = 4.0     # distancia do neutro ate cada extremo (1 e 9)

# --------------------------------------------------------------- parametros --
# |polaridade| minima para a palavra contar. O ANEW da nota media a palavras
# como "board" ou "cabinet", que ficam colados no neutro: elas nao carregam
# opiniao, so ruido. 0.30 equivale a ~1.2 ponto de distancia do neutro no SAM.
LIMIAR_NEUTRALIDADE = 0.30

# Quanto o arousal mexe no peso da palavra. Com 0.25, uma palavra calma pesa
# 0.75x e uma agitada 1.25x. Nunca inverte o sinal - o artigo trata arousal
# como intensidade (calmo <-> agitado), e nao como polaridade: "anger" e
# "affection" tem arousal parecido e polaridades opostas.
# Medido neste corpus: ligar ou desligar o arousal muda a acuracia balanceada
# em 0.1 ponto (65.6% -> 65.5%). Ele fica porque e o uso correto da dimensao
# segundo o artigo, mas nao e ele que faz o modelo funcionar - quem carrega o
# resultado sao a stoplist, a lematizacao, a negacao e o peso do titulo.
W_AROUSAL = 0.25

# O titulo da review e curto e quase sempre opiniao pura ("A fail, for several
# reasons"), enquanto o corpo mistura opiniao com resumo do enredo. Contar o
# titulo 2x aproveita essa densidade sem precisar de regra nova.
PESO_TITULO = 2

# O ANEW da nota a palavra isolada, fora de contexto: "not good" tem que
# deixar de ser positivo. A negacao inverte a polaridade das proximas
# JANELA_NEGACAO palavras, com desconto - negar nao e tao forte quanto afirmar
# o contrario ("not good" nao chega a "bad").
JANELA_NEGACAO = 3
FORCA_NEGACAO = 0.8

# Descartar o vocabulario de enredo (ver STOPLIST_ENREDO).
USAR_STOPLIST_ENREDO = True

# Reduzir flexoes nao encontradas ("loved" -> "love"). O ANEW lista sobretudo
# formas base, e o corpus e texto corrido.
USAR_LEMATIZACAO_SIMPLES = True

# Corte da decisao. 0.0 = massa agradavel iguala massa desagradavel, ou seja,
# o neutro do proprio SAM. Mantido em 0.0 para a decisao sair do artigo e nao
# do gabarito; o relatorio no fim mostra qual corte o rating premiaria.
#
# Trade-off, com os numeros deste corpus:
#   0.0  -> acuracia 75.4%, balanceada 62.6%. Nao supera o chute na maioria
#           (77.2%), porque resenha de cinema usa palavra agradavel mesmo para
#           detonar o filme: a media do score ja e +0.06 nas notas 1.
#   0.37 -> balanceada 65.6%, o melhor corte possivel, mas escolhido olhando o
#           rating - deixa de ser "so ANEW" e passa a ser calibrado no gabarito.
# Trocar por 0.37 e uma decisao legitima, desde que a apresentacao diga que o
# corte foi calibrado.
LIMIAR_DECISAO = 0.0

ROTULO_POS, ROTULO_NEG = "positivo", "negativo"

# Grava tambem 'anew_score' e 'anew_termos'. O score e o que vai virar figura
# na apresentacao, e anew_termos mostra em quantas palavras a decisao se
# apoiou. Deixe False para a saida ter exatamente uma coluna nova.
COLUNAS_DIAGNOSTICO = True

# --------------------------------------------------------------- stoplist ----
# O ANEW pontua a palavra, nao o seu papel na frase. Numa review da Odisseia,
# "war", "death", "sea" e "god" descrevem o *enredo*, e "actor" e "screen" o
# *meio* - nenhuma delas diz o que a pessoa achou do filme, mas todas tem
# pleasure longe do neutro e por isso dominariam o score. Esta e a mesma
# stoplist usada em scripts.py, mantida aqui para o script ficar autocontido.
STOPLIST_ENREDO = {
    # o meio: cinema, livro, teatro
    "movie", "film", "cinema", "theater", "screen", "star", "actor", "actress",
    "book", "story", "chapter", "author", "art", "scene", "part", "kind",
    # o enredo: guerra, mar, viagem
    "war", "army", "soldier", "weapon", "sword", "ship", "sea", "ocean", "water",
    "island", "storm", "wind", "home", "house", "journey", "adventure",
    # o enredo: mito e realeza
    "god", "goddess", "spirit", "hero", "king", "queen", "prince", "master",
    # os personagens
    "man", "woman", "wife", "husband", "son", "father", "mother", "family",
    "child", "boy", "girl", "people", "person", "friend", "stranger",
    # producao e tempo narrativo
    "music", "song", "voice", "time", "hour", "year", "day", "night",
    # abstratos que aparecem em qualquer resumo de enredo
    "life", "death", "world", "history", "money", "opinion", "thought",
    "mind", "heart", "body", "blood", "fire", "dark", "light",
}

NEGADORES = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "hardly", "barely", "scarcely", "rarely", "cannot", "cant",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont",
    "wouldnt", "shouldnt", "couldnt", "aint", "lacks", "lacking",
}

# Sufixos testados na ordem, sempre com o candidato validado contra o ANEW.
# "less" esta fora de proposito: "hopeless" -> "hope" inverteria o sentido.
SUFIXOS = (
    ("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
    ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", ""),
)


# ----------------------------------------------------------------- lexico ----
def carregar_anew(caminho: str = ANEW_CSV) -> dict[str, tuple[float, float]]:
    """Le o ANEW e devolve {termo: (polaridade, peso_arousal)}.

    polaridade  em [-1, +1], negativo = desagradavel, 0 = neutro do SAM.
    peso_arousal multiplicador >= 0 tirado do arousal; nunca muda o sinal.
    """
    lexico: dict[str, tuple[float, float]] = {}
    lidos = desc_neutro = desc_enredo = 0

    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for linha in csv.DictReader(f):
            termo = str(linha[COL_TERMO]).strip().lower()
            if not termo:
                continue
            lidos += 1

            if USAR_STOPLIST_ENREDO and termo in STOPLIST_ENREDO:
                desc_enredo += 1
                continue

            # de volta para a escala 1-9 do artigo, depois centrado no neutro
            pleasure = float(linha[COL_PLEASURE]) * FATOR_PARA_SAM
            arousal = float(linha[COL_AROUSAL]) * FATOR_PARA_SAM

            polaridade = (pleasure - SAM_NEUTRO) / SAM_AMPLITUDE
            if abs(polaridade) < LIMIAR_NEUTRALIDADE:
                desc_neutro += 1
                continue

            intensidade = (arousal - SAM_NEUTRO) / SAM_AMPLITUDE   # [-1, +1]
            lexico[termo] = (polaridade, 1.0 + W_AROUSAL * intensidade)

    print(f"[lexico] {lidos:,} termos lidos | {len(lexico):,} usaveis | "
          f"{desc_neutro:,} descartados por neutralidade | "
          f"{desc_enredo:,} por serem vocabulario de enredo")
    return lexico


# ---------------------------------------------------------------- scoring ----
def tokenizar(texto: str) -> list[str]:
    """Minusculas, "n't" virando " not" (senao a negacao se perde) e fora
    tudo que nao for letra."""
    texto = str(texto).lower().replace("n't", " not")
    return re.sub(r"[^a-z\s]", " ", texto).split()


def buscar(palavra: str, lexico: dict) -> tuple[float, float] | None:
    """Busca exata; se falhar e a lematizacao estiver ligada, tenta tirar
    sufixos comuns e so aceita o resultado se ele existir no ANEW."""
    entrada = lexico.get(palavra)
    if entrada is not None or not USAR_LEMATIZACAO_SIMPLES:
        return entrada

    for sufixo, troca in SUFIXOS:
        if palavra.endswith(sufixo):
            base = palavra[: -len(sufixo)] + troca
            if len(base) > 2 and base in lexico:
                return lexico[base]
    return None


def pontuar(titulo: str, texto: str, lexico: dict) -> tuple[float, int]:
    """Score em [-1, +1] e quantas palavras do ANEW foram encontradas.

    O score e um indice de polaridade - massa agradavel menos massa
    desagradavel, dividido pela massa total. Dividir pela massa faz o
    resultado nao depender do tamanho da review: uma review longa e morna
    nao vira "muito positiva" so por acumular palavras.
    """
    palavras = tokenizar(titulo) * PESO_TITULO + tokenizar(texto)

    soma = massa = 0.0
    encontradas = 0
    desde_negador = JANELA_NEGACAO + 1

    for palavra in palavras:
        if palavra in NEGADORES:
            desde_negador = 0
            continue
        desde_negador += 1

        entrada = buscar(palavra, lexico)
        if entrada is None:
            continue

        polaridade, peso_arousal = entrada
        if desde_negador <= JANELA_NEGACAO:
            polaridade = -polaridade * FORCA_NEGACAO

        contribuicao = polaridade * peso_arousal
        soma += contribuicao
        massa += abs(contribuicao)
        encontradas += 1

    if massa == 0.0:
        return 0.0, 0
    return soma / massa, encontradas


def classificar(score: float) -> str:
    """Rotulo binario. Reviews sem nenhuma palavra do ANEW tem score 0.0 e
    caem em "positivo" por causa do >=; sao poucas e o relatorio as conta."""
    return ROTULO_POS if score >= LIMIAR_DECISAO else ROTULO_NEG


# -------------------------------------------------------------- avaliacao ----
# O corpus nao tem rotulo de sentimento: o 'rating' de 1 a 10 que o autor deu
# e usado *somente aqui*, para medir o acerto depois da classificacao. Ele nao
# entra em nenhuma etapa da decisao.
def rotulo_do_rating(nota: float) -> str:
    return ROTULO_POS if nota >= 6 else ROTULO_NEG


def metricas(pares: list[tuple[str, str]]) -> dict:
    """pares = [(real, previsto), ...]"""
    total = len(pares)
    acertos = sum(1 for real, prev in pares if real == prev)
    saida = {"n": total, "acuracia": acertos / total if total else 0.0}

    recalls = []
    for classe in (ROTULO_NEG, ROTULO_POS):
        vp = sum(1 for r, p in pares if p == classe and r == classe)
        fp = sum(1 for r, p in pares if p == classe and r != classe)
        fn = sum(1 for r, p in pares if p != classe and r == classe)
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        saida[classe] = {"precisao": prec, "recall": rec, "f1": f1,
                         "suporte": vp + fn}
        recalls.append(rec)

    saida["acc_balanceada"] = sum(recalls) / 2
    return saida


def melhor_limiar(dados: list[tuple[float, str]]) -> tuple[float, float]:
    """Corte que maximizaria a acuracia balanceada. Serve de diagnostico: diz
    se o score ja separa as classes e falta so calibrar o corte."""
    melhor, melhor_acc = LIMIAR_DECISAO, -1.0
    passo = 0.01
    corte = -1.0
    while corte <= 1.0:
        pares = [(real, ROTULO_POS if s >= corte else ROTULO_NEG)
                 for s, real in dados]
        acc = metricas(pares)["acc_balanceada"]
        if acc > melhor_acc:
            melhor, melhor_acc = corte, acc
        corte += passo
    return melhor, melhor_acc


def relatorio(linhas: list[dict]) -> None:
    print("\n" + "=" * 64)
    print(" RESULTADO DA CLASSIFICACAO")
    print("=" * 64)

    total = len(linhas)
    pos = sum(1 for l in linhas if l["sentimento"] == ROTULO_POS)
    sem_termo = sum(1 for l in linhas if l["anew_termos"] == 0)
    media_termos = sum(l["anew_termos"] for l in linhas) / total
    media_palavras = sum(int(l["n_palavras"] or 0) for l in linhas) / total

    print(f"\nReviews classificadas:     {total:,}")
    print(f"  positivo:                {pos:,} ({pos / total:.1%})")
    print(f"  negativo:                {total - pos:,} ({1 - pos / total:.1%})")
    print(f"Palavras do ANEW por review: {media_termos:.1f} de "
          f"{media_palavras:.0f} ({media_termos / media_palavras:.1%} de cobertura)")
    print(f"Reviews sem nenhuma palavra do ANEW: {sem_termo}")

    print("\nDistribuicao do score (cada faixa de 0.1):")
    for i in range(-10, 10):
        lo, hi = i / 10, (i + 1) / 10
        n = sum(1 for l in linhas if lo <= l["anew_score"] < hi)
        if n:
            print(f"  {lo:+.1f} a {hi:+.1f} | {n:>5} {'#' * max(1, n * 50 // total)}")

    # ---- avaliacao contra o rating (gabarito, nunca usado na decisao) ----
    com_nota = [l for l in linhas if str(l["rating"]).strip()]
    if not com_nota:
        return

    pares = [(rotulo_do_rating(float(l["rating"])), l["sentimento"])
             for l in com_nota]
    m = metricas(pares)
    base = max(sum(1 for r, _ in pares if r == ROTULO_POS),
               sum(1 for r, _ in pares if r == ROTULO_NEG)) / len(pares)

    print("\n" + "-" * 64)
    print(f" Avaliacao contra o rating do autor (>= 6 = positivo), n = {m['n']:,}")
    print("-" * 64)
    print(f"Acuracia:               {m['acuracia']:.1%}")
    print(f"Chute na maioria:       {base:.1%}   <- so e util se superar isso")
    print(f"Acuracia balanceada:    {m['acc_balanceada']:.1%}")
    for classe in (ROTULO_NEG, ROTULO_POS):
        c = m[classe]
        print(f"  {classe:<9} precisao={c['precisao']:.1%} "
              f"recall={c['recall']:.1%} f1={c['f1']:.3f} n={c['suporte']:,}")

    print("\nMatriz de confusao:")
    print(f"  {'':<12}{'prev. neg':>11}{'prev. pos':>11}")
    for real in (ROTULO_NEG, ROTULO_POS):
        n_neg = sum(1 for r, p in pares if r == real and p == ROTULO_NEG)
        n_pos = sum(1 for r, p in pares if r == real and p == ROTULO_POS)
        print(f"  real {real:<7}{n_neg:>11,}{n_pos:>11,}")

    print("\nScore medio por rating (tem que subir de 1 para 10):")
    for nota in range(1, 11):
        sub = [l["anew_score"] for l in com_nota if float(l["rating"]) == nota]
        if sub:
            media = sum(sub) / len(sub)
            print(f"  {nota:>2} | n={len(sub):>5} media={media:+.3f} "
                  f"{'#' * max(1, int(30 * (media + 1) / 2))}")

    corte, acc = melhor_limiar(
        [(l["anew_score"], rotulo_do_rating(float(l["rating"]))) for l in com_nota])
    print(f"\nLIMIAR_DECISAO em uso: {LIMIAR_DECISAO:+.2f}")
    print(f"Melhor corte possivel: {corte:+.2f} -> acuracia balanceada "
          f"{acc:.1%} (diagnostico; mudar isso e calibrar pelo gabarito)")


# ------------------------------------------------------------------- main ----
def main() -> None:
    # reviews longas passam do limite padrao de campo do modulo csv
    csv.field_size_limit(10 ** 7)

    lexico = carregar_anew()

    with open(REVIEWS_CSV, encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        colunas = list(leitor.fieldnames or [])
        linhas = list(leitor)

    if not linhas:
        sys.exit(f"[erro] '{REVIEWS_CSV}' esta vazio.")
    print(f"[corpus] {len(linhas):,} reviews lidas de {REVIEWS_CSV}")

    for linha in linhas:
        score, n_termos = pontuar(linha.get("titulo_review") or "",
                                  linha.get("texto") or "", lexico)
        linha["anew_score"] = round(score, 4)
        linha["anew_termos"] = n_termos
        linha["sentimento"] = classificar(score)

    novas = ["anew_score", "anew_termos", "sentimento"] if COLUNAS_DIAGNOSTICO \
        else ["sentimento"]

    # extrasaction="ignore" descarta as colunas de diagnostico na escrita sem
    # tirar elas das linhas em memoria, que o relatorio ainda usa.
    with open(SAIDA_CSV, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas + novas,
                                  extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)

    relatorio(linhas)
    print(f"\nSaida: {SAIDA_CSV} "
          f"({len(linhas):,} linhas, {len(colunas) + len(novas)} colunas)")
    print(f"Colunas novas: {', '.join(novas)}")


if __name__ == "__main__":
    main()
