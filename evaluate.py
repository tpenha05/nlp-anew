# -*- coding: utf-8 -*-
"""
Avaliacao do scoring ANEW: mede o quanto acerta e onde perde acuracia.

Roda no mesmo diretorio de anew_sentiment_v2.py.

Cinco blocos:
  1. Metricas gerais, com limiares otimizados (separa qualidade do score
     da qualidade do corte).
  2. Ablacoes: liga/desliga cada componente e mostra o delta.
  3. Erro por faixa: cobertura, tamanho da review, rating.
  4. Termos que puxam o erro - candidatos a stoplist.
  5. Piores casos individuais.
"""

from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pandas as pd

import scripts as base

REVIEWS_CSV = "data/imdb_odyssey_reviews.csv"
CLASSES = ["negativo", "neutro", "positivo"]


def rotulo_real(nota: float) -> str:
    return "positivo" if nota >= 7 else ("negativo" if nota <= 4 else "neutro")


def metricas(real: pd.Series, pred: pd.Series) -> dict:
    f1s, recalls = [], []
    for c in CLASSES:
        tp = ((pred == c) & (real == c)).sum()
        fp = ((pred == c) & (real != c)).sum()
        fn = ((pred != c) & (real == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        recalls.append(rec)
    return {
        "acuracia": (pred == real).mean(),
        "acc_balanceada": sum(recalls) / len(CLASSES),
        "f1_macro": sum(f1s) / len(CLASSES),
        "f1_negativo": f1s[0],
    }


def melhores_limiares(scores: pd.Series, real: pd.Series) -> tuple[float, float, dict]:
    """Busca o par de cortes que maximiza F1 macro. Da o teto do score atual."""
    s = scores.to_numpy()
    r = real.to_numpy()
    grade = np.quantile(s, np.arange(1, 20) / 20)
    melhor = (0.0, 0.0, {"f1_macro": -1.0})
    for i, neg in enumerate(grade):
        for pos in grade[i + 1:]:
            pred = np.where(s >= pos, "positivo",
                            np.where(s <= neg, "negativo", "neutro"))
            m = metricas(pd.Series(r), pd.Series(pred))
            if m["f1_macro"] > melhor[2]["f1_macro"]:
                melhor = (float(neg), float(pos), m)
    return melhor


def avaliar_config(reviews: pd.DataFrame, **config) -> dict:
    for chave, valor in config.items():
        setattr(base, chave, valor)
    lexico = base.carregar_lexico(base.ANEW_CSV)

    titulos = reviews.get("titulo_review", pd.Series([""] * len(reviews))).fillna("")
    linhas = [base.pontuar(t, ti, lexico)
              for t, ti in zip(reviews["texto"].fillna(""), titulos)]
    df = pd.concat([reviews.reset_index(drop=True), pd.DataFrame(linhas)], axis=1)

    v = df.dropna(subset=["rating"])
    v = v[v["n_termos"] >= base.MIN_TERMOS]
    real = v["rating"].apply(rotulo_real)

    neg, pos, m = melhores_limiares(v["score"].reset_index(drop=True),
                                    real.reset_index(drop=True))
    m["spearman"] = v["score"].rank().corr(v["rating"].rank())
    m["cobertura"] = v["cobertura"].mean()
    m["avaliadas"] = len(v)
    m["limiares"] = (round(float(neg), 3), round(float(pos), 3))
    return {"metricas": m, "df": df, "real": real, "limiares": (neg, pos)}


# ---------------------------------------------------------------- blocos -----
def bloco_geral(res: dict) -> None:
    m = res["metricas"]
    v = res["df"].dropna(subset=["rating"])
    v = v[v["n_termos"] >= base.MIN_TERMOS]
    real = res["real"]
    maioria = real.value_counts(normalize=True).max()

    print("\n" + "=" * 62)
    print(" 1. DESEMPENHO GERAL (limiares otimizados)")
    print("=" * 62)
    print(f"Reviews avaliadas:        {m['avaliadas']:,}")
    print(f"Cobertura media:          {m['cobertura']:.1%}")
    print(f"Spearman x rating:        {m['spearman']:+.3f}   <- qualidade do score")
    print(f"Limiares otimos:          {m['limiares']}")
    print(f"Acuracia:                 {m['acuracia']:.1%}")
    print(f"Linha de base (maioria):  {maioria:.1%}   <- so bate isso se for util")
    print(f"Acuracia balanceada:      {m['acc_balanceada']:.1%}")
    print(f"F1 macro:                 {m['f1_macro']:.3f}")
    print(f"F1 da classe negativa:    {m['f1_negativo']:.3f}   <- a classe dificil")

    neg, pos = res["limiares"]
    pred = v["score"].apply(
        lambda s: "positivo" if s >= pos else ("negativo" if s <= neg else "neutro"))
    print("\nMatriz de confusao (linha = rating, coluna = predito):")
    print(pd.crosstab(real, pred))

    print("\nScore medio por rating (tem que subir monotonicamente):")
    for nota, sub in v.groupby("rating"):
        barra = "#" * max(1, int(30 * (sub["score"].mean() + 1) / 2))
        print(f"  {int(nota):>2} | n={len(sub):>5} media={sub['score'].mean():+.3f} {barra}")


def bloco_ablacoes(reviews: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print(" 2. ABLACOES (cada linha muda um componente)")
    print("=" * 62)

    configs = {
        "atual": {},
        "sem stoplist": {"USAR_STOPLIST": False},
        "com lexico extra": {"USAR_LEXICO_EXTRA": True},
        "stoplist + extra": {"USAR_STOPLIST": True, "USAR_LEXICO_EXTRA": True},
        "sem negacao": {"JANELA_NEGACAO": 0},
        "sem peso no titulo": {"PESO_TITULO": 0},
        "titulo x4": {"PESO_TITULO": 4},
        "so valencia": {"W_VALENCIA": 1.0, "W_AROUSAL": 0.0},
        "arousal 0.5": {"W_VALENCIA": 0.5, "W_AROUSAL": 0.5},
        "limiar sent 0.10": {"LIMIAR_SENTIMENTO": 0.10},
        "limiar sent 0.50": {"LIMIAR_SENTIMENTO": 0.50},
    }
    padrao = {k: getattr(base, k) for k in
              ("USAR_STOPLIST", "USAR_LEXICO_EXTRA", "JANELA_NEGACAO",
               "PESO_TITULO", "W_VALENCIA", "W_AROUSAL", "LIMIAR_SENTIMENTO")}

    print(f"\n{'config':<20} {'spearman':>9} {'f1_macro':>9} {'f1_neg':>8} {'cobert':>8}")
    print("-" * 58)
    for nome, mudanca in configs.items():
        with contextlib.redirect_stdout(io.StringIO()):
            res = avaliar_config(reviews, **{**padrao, **mudanca})
        m = res["metricas"]
        print(f"{nome:<20} {m['spearman']:>+9.3f} {m['f1_macro']:>9.3f} "
              f"{m['f1_negativo']:>8.3f} {m['cobertura']:>7.1%}")
    for k, v in padrao.items():
        setattr(base, k, v)


def bloco_faixas(res: dict) -> None:
    v = res["df"].dropna(subset=["rating"]).copy()
    v = v[v["n_termos"] >= base.MIN_TERMOS]

    print("\n" + "=" * 62)
    print(" 3. ONDE O SINAL E MAIS FORTE")
    print("=" * 62)

    print("\nPor numero de termos encontrados:")
    for lo, hi in [(3, 5), (5, 10), (10, 20), (20, 40), (40, 10000)]:
        s = v[(v["n_termos"] >= lo) & (v["n_termos"] < hi)]
        if len(s) > 30:
            rho = s["score"].rank().corr(s["rating"].rank())
            print(f"  {lo:>3}-{hi if hi < 999 else '+':<5} n={len(s):>5}  spearman={rho:+.3f}")

    print("\nPor tamanho da review (palavras):")
    for lo, hi in [(0, 100), (100, 250), (250, 500), (500, 100000)]:
        s = v[(v["n_palavras"] >= lo) & (v["n_palavras"] < hi)]
        if len(s) > 30:
            rho = s["score"].rank().corr(s["rating"].rank())
            print(f"  {lo:>4}-{hi if hi < 99999 else '+':<6} n={len(s):>5}  spearman={rho:+.3f}")


def bloco_termos(res: dict) -> None:
    v = res["df"].dropna(subset=["rating"]).copy()
    v = v[v["n_termos"] >= base.MIN_TERMOS]
    neg, pos = res["limiares"]

    def contar(sub: pd.DataFrame) -> dict[str, int]:
        c: dict[str, int] = {}
        for linha in sub["top_termos"].dropna():
            for m in re.finditer(r"(\w+)\(", linha):
                c[m.group(1)] = c.get(m.group(1), 0) + 1
        return c

    sub_fp = v[(v["score"] >= pos) & (v["rating"] <= 4)]
    sub_fn = v[(v["score"] <= neg) & (v["rating"] >= 7)]
    falso_pos, falso_neg, total = contar(sub_fp), contar(sub_fn), contar(v)
    n_fp, n_fn = max(1, len(sub_fp)), max(1, len(sub_fn))

    print("\n" + "=" * 62)
    print(" 4. TERMOS QUE PUXAM O ERRO (candidatos a stoplist)")
    print("=" * 62)

    def ranking(erro: dict[str, int], n_erro: int, titulo: str) -> None:
        print(f"\n{titulo}  (n={n_erro})")
        if not erro:
            print("  (nenhum caso nesta categoria)")
            return
        linhas = []
        for termo, qtd in erro.items():
            if qtd < 5:
                continue
            taxa_erro = qtd / n_erro
            taxa_geral = total.get(termo, 1) / len(v)
            linhas.append((taxa_erro / max(taxa_geral, 1e-9), termo, qtd))
        for lift, termo, qtd in sorted(linhas, reverse=True)[:12]:
            print(f"  {termo:<16} aparece {qtd:>4}x  lift={lift:.2f}")

    ranking(falso_pos, n_fp, "Em reviews que o modelo achou positivas mas tem rating <= 4:")
    ranking(falso_neg, n_fn, "Em reviews que o modelo achou negativas mas tem rating >= 7:")


def bloco_piores(res: dict, n: int = 5) -> None:
    v = res["df"].dropna(subset=["rating"]).copy()
    v = v[v["n_termos"] >= base.MIN_TERMOS]
    v["erro"] = (v["rating"] - 5.5) / 4.5 - v["score"]

    print("\n" + "=" * 62)
    print(" 5. PIORES CASOS")
    print("=" * 62)
    for titulo, sub in [("Rating alto, score baixo:", v.nlargest(n, "erro")),
                        ("Rating baixo, score alto:", v.nsmallest(n, "erro"))]:
        print(f"\n{titulo}")
        for _, r in sub.iterrows():
            print(f"  rating={int(r['rating'])} score={r['score']:+.3f} | "
                  f"{str(r['texto'])[:90]}...")
            print(f"    termos: {r['top_termos']}")


def main() -> None:
    reviews = pd.read_csv(REVIEWS_CSV)
    print(f"Carregado: {len(reviews):,} reviews")

    res = avaliar_config(reviews)
    bloco_geral(res)
    bloco_faixas(res)
    bloco_termos(res)
    bloco_piores(res)
    bloco_ablacoes(reviews)

    print("\n" + "=" * 62)
    print(" COMO LER ISSO")
    print("=" * 62)
    print("Spearman mede o score; F1 macro mede o score + os cortes.")
    print("Spearman baixo com F1 alto = sorte nos cortes.")
    print("Spearman alto com F1 baixo = so falta calibrar.")
    print("Se a acuracia nao supera a linha de base da maioria, o modelo")
    print("ainda nao esta ganhando nada de um chute constante.")


if __name__ == "__main__":
    main()