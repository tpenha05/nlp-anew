# -*- coding: utf-8 -*-
"""
Sentiment scoring ANEW - reviews de "The Odyssey" (2026).

Score = indice de polaridade: (massa_positiva - massa_negativa) / (massa_total).
Palavras neutras e vocabulario de enredo sao descartadas antes da conta.
"""

from __future__ import annotations

import re

import pandas as pd

# ------------------------------------------------------------------ config ---
ANEW_CSV = "data/anew.csv"
REVIEWS_CSV = "data/imdb_odyssey_reviews.csv"
SAIDA_CSV = "imdb_odyssey_scores_v2.csv"

COL_TERM, COL_VAL, COL_ARO = "term", "pleasure", "arousal"
CENTRO, MEIA_FAIXA = 50.0, 50.0        # escala 0-100

W_VALENCIA, W_AROUSAL = 0.75, 0.25
LIMIAR_SENTIMENTO = 0.30               # |v_norm| minimo para o termo contar
PESO_TITULO = 2

USAR_LEXICO_EXTRA = False              # True = deixa de ser ANEW puro
USAR_STOPLIST = True

LIMIAR_POS, LIMIAR_NEG = 0.30, 0.00
MIN_TERMOS = 3

# Vocabulario do enredo da Odisseia e termos genericos de cinema: aparecem
# muito e nao carregam opiniao.
STOPLIST_ENREDO = {
    "movie", "film", "cinema", "theater", "screen", "star", "actor", "actress",
    "book", "story", "chapter", "author", "art", "scene", "part", "kind",
    "war", "army", "soldier", "weapon", "sword", "ship", "sea", "ocean", "water",
    "island", "storm", "wind", "home", "house", "journey", "adventure",
    "god", "goddess", "spirit", "hero", "king", "queen", "prince", "master",
    "man", "woman", "wife", "husband", "son", "father", "mother", "family",
    "child", "boy", "girl", "people", "person", "friend", "stranger",
    "music", "song", "voice", "time", "hour", "year", "day", "night",
    "life", "death", "world", "history", "money", "opinion", "thought",
    "mind", "heart", "body", "blood", "fire", "dark", "light",
}

# Vocabulario de critica que o ANEW nao cobre. (pleasure, arousal) em 0-100.
LEXICO_EXTRA = {
    "masterpiece": (92, 70), "brilliant": (89, 68), "stunning": (88, 74),
    "gorgeous": (88, 65), "flawless": (90, 60), "breathtaking": (90, 78),
    "compelling": (78, 62), "gripping": (80, 72), "immersive": (80, 62),
    "riveting": (82, 74), "captivating": (84, 66), "spectacular": (87, 75),
    "remarkable": (83, 60), "unforgettable": (86, 68), "superb": (88, 62),
    "impeccable": (86, 55), "atmospheric": (72, 50), "epic": (78, 66),
    "moving": (76, 58), "touching": (78, 55), "clever": (76, 55),
    "solid": (68, 45), "decent": (63, 40), "underrated": (70, 50),
    "boring": (16, 20), "dull": (18, 18), "tedious": (16, 22),
    "bland": (24, 25), "forgettable": (25, 28), "predictable": (28, 30),
    "overrated": (22, 45), "pretentious": (22, 48), "shallow": (26, 35),
    "clumsy": (25, 40), "messy": (24, 42), "mess": (18, 50),
    "disappointing": (15, 45), "disappointment": (14, 45), "flawed": (30, 40),
    "awful": (8, 65), "terrible": (8, 68), "horrible": (7, 70),
    "garbage": (7, 60), "trash": (8, 60), "waste": (10, 55),
    "pointless": (18, 40), "lazy": (22, 30), "bloated": (25, 35),
    "rushed": (30, 55), "confusing": (30, 50), "incoherent": (20, 48),
    "wooden": (28, 25), "cringe": (15, 55), "generic": (32, 28),
    "pacing": (45, 40), "miscast": (22, 45), "melodramatic": (32, 50),
}

NEGADORES = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "hardly", "barely", "scarcely", "rarely", "cannot", "cant",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont",
    "wouldnt", "shouldnt", "couldnt", "aint", "lacks", "lacking",
}
JANELA_NEGACAO = 3

INTENSIFICADORES = {
    "very": 1.5, "extremely": 1.8, "incredibly": 1.8, "absolutely": 1.7,
    "totally": 1.5, "utterly": 1.7, "truly": 1.4, "really": 1.3,
    "so": 1.3, "completely": 1.6, "deeply": 1.5, "highly": 1.4,
    "slightly": 0.6, "somewhat": 0.6, "fairly": 0.7, "mildly": 0.6,
    "kinda": 0.7, "mostly": 0.9,
}


# ------------------------------------------------------------------ lexico ---
def carregar_lexico(caminho: str = ANEW_CSV) -> dict[str, dict]:
    df = pd.read_csv(caminho)
    df[COL_TERM] = df[COL_TERM].astype(str).str.strip().str.lower()

    bruto = {t: (float(v), float(a)) for t, v, a
             in zip(df[COL_TERM], df[COL_VAL], df[COL_ARO])}
    if USAR_LEXICO_EXTRA:
        bruto.update({t.lower(): (float(v), float(a))
                      for t, (v, a) in LEXICO_EXTRA.items()})

    lexico: dict[str, dict] = {}
    descartados = 0
    for termo, (valencia, arousal) in bruto.items():
        if USAR_STOPLIST and termo in STOPLIST_ENREDO:
            descartados += 1
            continue

        v_norm = (valencia - CENTRO) / MEIA_FAIXA
        a_norm = (arousal - CENTRO) / MEIA_FAIXA
        if abs(v_norm) < LIMIAR_SENTIMENTO:
            descartados += 1
            continue

        lexico[termo] = {
            "v_norm": v_norm,
            "a_norm": a_norm,
            "peso": min(abs(v_norm), 1.0),
            "pleasure": valencia,
        }

    print(f"[info] {len(bruto):,} termos lidos | {len(lexico):,} usaveis | "
          f"{descartados:,} descartados (neutros ou enredo)")
    return lexico


# ----------------------------------------------------------------- scoring ---
def tokenizar(texto: str) -> list[str]:
    texto = str(texto).lower().replace("n't", " not")
    return re.sub(r"[^a-z\s]", " ", texto).split()


def pontuar(texto: str, titulo: str, lexico: dict[str, dict]) -> dict:
    palavras = tokenizar(titulo) * PESO_TITULO + tokenizar(texto)

    massa_pos = massa_neg = 0.0
    contribuicoes: list[tuple[str, float]] = []
    dist_negador = 99
    multiplicador = 1.0

    for palavra in palavras:
        if palavra in NEGADORES:
            dist_negador = 0
            continue
        if palavra in INTENSIFICADORES:
            multiplicador = INTENSIFICADORES[palavra]
            continue
        dist_negador += 1

        entrada = lexico.get(palavra)
        if entrada is None:
            multiplicador = 1.0
            continue

        v = entrada["v_norm"]
        if dist_negador <= JANELA_NEGACAO:
            v = -v * 0.8

        sinal = 1.0 if v >= 0 else -1.0
        s = W_VALENCIA * v + W_AROUSAL * sinal * entrada["a_norm"]
        contrib = entrada["peso"] * s * multiplicador
        multiplicador = 1.0

        if contrib >= 0:
            massa_pos += contrib
        else:
            massa_neg += -contrib
        contribuicoes.append((palavra, contrib))

    total = massa_pos + massa_neg
    if total == 0:
        return {"score": 0.0, "n_termos": 0, "cobertura": 0.0, "top_termos": ""}

    contribuicoes.sort(key=lambda x: abs(x[1]), reverse=True)
    return {
        "score": (massa_pos - massa_neg) / total,
        "n_termos": len(contribuicoes),
        "cobertura": len(contribuicoes) / len(palavras),
        "top_termos": ", ".join(f"{p}({c:+.2f})" for p, c in contribuicoes[:5]),
    }


def rotular(score: float, n_termos: int) -> str:
    if n_termos < MIN_TERMOS:
        return "indefinido"
    if score >= LIMIAR_POS:
        return "positivo"
    if score <= LIMIAR_NEG:
        return "negativo"
    return "neutro"


def analisar(df: pd.DataFrame, lexico: dict[str, dict]) -> pd.DataFrame:
    titulos = df.get("titulo_review", pd.Series([""] * len(df))).fillna("")
    linhas = [pontuar(t, ti, lexico)
              for t, ti in zip(df["texto"].fillna(""), titulos)]
    saida = pd.concat([df.reset_index(drop=True), pd.DataFrame(linhas)], axis=1)
    saida["rotulo"] = [rotular(l["score"], l["n_termos"]) for l in linhas]
    return saida


# --------------------------------------------------------------- avaliacao ---
def avaliar(df: pd.DataFrame) -> None:
    v = df.dropna(subset=["rating"])
    v = v[v["n_termos"] >= MIN_TERMOS]
    if v.empty:
        print("[aviso] nada a avaliar")
        return

    real = v["rating"].apply(
        lambda n: "positivo" if n >= 7 else ("negativo" if n <= 4 else "neutro"))
    pred = v["rotulo"]
    classes = ["negativo", "neutro", "positivo"]

    f1s, recalls = [], []
    for c in classes:
        tp = ((pred == c) & (real == c)).sum()
        fp = ((pred == c) & (real != c)).sum()
        fn = ((pred != c) & (real == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        recalls.append(rec)

    rho = v["score"].rank().corr(v["rating"].rank())
    maioria = real.value_counts(normalize=True).max()

    print("\n" + "=" * 60)
    print(f"Reviews avaliadas:      {len(v):,}")
    print(f"Spearman x rating:      {rho:+.3f}")
    print(f"Acuracia:               {(pred == real).mean():.1%}")
    print(f"Linha de base (maioria):{maioria:.1%}")
    print(f"Acuracia balanceada:    {sum(recalls) / 3:.1%}")
    print(f"F1 macro:               {sum(f1s) / 3:.3f}")
    print(f"Cobertura media:        {v['cobertura'].mean():.1%}")
    print("\nMatriz de confusao (linha = rating, coluna = predito):")
    print(pd.crosstab(real, pred))


def calibrar(df: pd.DataFrame) -> None:
    v = df.dropna(subset=["rating"])
    if v.empty:
        return
    p_neg = (v["rating"] <= 4).mean()
    p_pos = (v["rating"] >= 7).mean()
    print(f"\nCalibracao por quantis: LIMIAR_NEG = "
          f"{df['score'].quantile(p_neg):+.3f} | LIMIAR_POS = "
          f"{df['score'].quantile(1 - p_pos):+.3f}")


def termos_dominantes(df: pd.DataFrame, n: int = 25) -> None:
    """Lista os termos que mais aparecem entre os de maior peso.
    Use para alimentar a STOPLIST_ENREDO."""
    contagem: dict[str, int] = {}
    for linha in df["top_termos"].dropna():
        for m in re.finditer(r"(\w+)\(", linha):
            contagem[m.group(1)] = contagem.get(m.group(1), 0) + 1
    print(f"\nTermos mais influentes no corpus:")
    for termo, qtd in sorted(contagem.items(), key=lambda x: -x[1])[:n]:
        print(f"  {termo:<16} {qtd}")


# ------------------------------------------------------------------- main ----
TESTES = [
    "I loved this movie, it was fantastic!",
    "I hated this movie, it was terrible.",
    "Nolan's Odyssey is not a masterpiece, just a beautiful bore.",
    "It never gets boring, the cinematography is stunning.",
    "A cruel, violent and brilliant adaptation of the epic.",
]


def main() -> None:
    lexico = carregar_lexico()

    for texto in TESTES:
        r = pontuar(texto, "", lexico)
        print(f"\n{texto}\n  score={r['score']:+.3f} -> "
              f"{rotular(r['score'], r['n_termos'])}  [{r['top_termos']}]")

    try:
        reviews = pd.read_csv(REVIEWS_CSV)
    except FileNotFoundError:
        print(f"\n[info] '{REVIEWS_CSV}' nao encontrado.")
        return

    resultado = analisar(reviews, lexico)
    resultado.to_csv(SAIDA_CSV, index=False, encoding="utf-8-sig")

    print(f"\n\n{len(resultado):,} reviews analisadas -> {SAIDA_CSV}")
    print(resultado["rotulo"].value_counts().to_string())
    calibrar(resultado)
    avaliar(resultado)
    termos_dominantes(resultado)


if __name__ == "__main__":
    main()