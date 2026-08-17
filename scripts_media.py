# -*- coding: utf-8 -*-
"""Baseline: media do pleasure das palavras do ANEW.

Dois experimentos independentes: um usando so o titulo da review, outro so o
texto. Com 81% de positivos, acuracia crua nao mede nada (prever "positivo"
sempre ja da 81%) - o que vale e a acuracia balanceada e o F1 da negativa.
"""

import re

import pandas as pd

ANEW_CSV = "data/anew.csv"
REVIEWS_CSV = "data/imdb_odyssey_reviews.csv"
SAIDA_CSV = "baseline_scores.csv"
SEMENTE = 42

anew = pd.read_csv(ANEW_CSV)
lexico = dict(zip(anew["term"].str.strip().str.lower(), anew["pleasure"]))


def media_pleasure(texto):
    palavras = re.sub(r"[^a-z\s]", " ", str(texto).lower()).split()
    valores = [lexico[p] for p in palavras if p in lexico]
    return sum(valores) / len(valores) if valores else None


def n_termos(texto):
    palavras = re.sub(r"[^a-z\s]", " ", str(texto).lower()).split()
    return sum(1 for p in palavras if p in lexico)


def avaliar(real, predito, titulo):
    tp = ((predito == "positivo") & (real == "positivo")).sum()
    fp = ((predito == "positivo") & (real == "negativo")).sum()
    tn = ((predito == "negativo") & (real == "negativo")).sum()
    fn = ((predito == "negativo") & (real == "positivo")).sum()

    rec_pos = tp / (tp + fn) if tp + fn else 0.0
    rec_neg = tn / (tn + fp) if tn + fp else 0.0
    prec_neg = tn / (tn + fn) if tn + fn else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if prec_neg + rec_neg else 0.0

    acuracia = (predito == real).mean()
    maioria = real.value_counts(normalize=True).max()

    print(f"\n  --- {titulo} ---")
    print(f"  Acuracia:              {acuracia:.1%}")
    print(f"  Chute na maioria:      {maioria:.1%}   (ganho: {acuracia - maioria:+.1%})")
    print(f"  Acuracia balanceada:   {(rec_pos + rec_neg) / 2:.1%}   (acaso = 50%)")
    print(f"  Recall positivo:       {rec_pos:.1%}")
    print(f"  Recall negativo:       {rec_neg:.1%}")
    print(f"  F1 da classe negativa: {f1_neg:.3f}")
    return {"experimento": titulo, "acuracia": acuracia,
            "acc_balanceada": (rec_pos + rec_neg) / 2,
            "recall_neg": rec_neg, "f1_neg": f1_neg}


def experimento(df, coluna, nome):
    print("\n" + "=" * 60)
    print(f" EXPERIMENTO: {nome}  (coluna '{coluna}')")
    print("=" * 60)

    d = df.copy()
    d["media"] = d[coluna].apply(media_pleasure)
    d["n_termos"] = d[coluna].apply(n_termos)

    v = d.dropna(subset=["media", "real"]).copy()
    descartadas = len(d.dropna(subset=["real"])) - len(v)
    mediana = v["media"].median()

    print(f"Avaliadas: {len(v):,}  |  sem palavra do ANEW: {descartadas:,}")
    print(f"Termos por review: mediana={v['n_termos'].median():.0f}  "
          f"media={v['n_termos'].mean():.1f}")
    print(f"Pleasure medio: {v['media'].mean():.2f}  |  mediana: {mediana:.2f}")

    v["pred_50"] = v["media"].apply(lambda m: "positivo" if m > 50 else "negativo")
    v["pred_mediana"] = v["media"].apply(
        lambda m: "positivo" if m > mediana else "negativo")

    linhas = [avaliar(v["real"], v["pred_50"], f"{nome} - corte em 50"),
              avaliar(v["real"], v["pred_mediana"], f"{nome} - corte na mediana")]

    n = v["real"].value_counts().min()
    bal = pd.concat([g.sample(n, random_state=SEMENTE) for _, g in v.groupby("real")])
    linhas.append(avaliar(bal["real"], bal["pred_mediana"],
                          f"{nome} - balanceado ({n} de cada)"))

    print("\n  Pleasure medio por rating:")
    print(v.groupby("rating")["media"].agg(["count", "mean"]).round(2).to_string())

    v = v.rename(columns={"media": f"media_{coluna}", "pred_50": f"pred50_{coluna}",
                          "pred_mediana": f"predmed_{coluna}",
                          "n_termos": f"ntermos_{coluna}"})
    return linhas, v


df = pd.read_csv(REVIEWS_CSV)
df["real"] = df["rating"].apply(
    lambda n: None if pd.isna(n) else ("positivo" if n >= 5 else "negativo"))

res_titulo, v_titulo = experimento(df, "titulo_review", "TITULO")
res_texto, v_texto = experimento(df, "texto", "TEXTO")

print("\n" + "=" * 60)
print(" COMPARACAO")
print("=" * 60)
resumo = pd.DataFrame(res_titulo + res_texto)
print(resumo.round(3).to_string(index=False))

saida = v_texto.join(
    v_titulo[["media_titulo_review", "ntermos_titulo_review",
              "predmed_titulo_review"]], how="left")
saida.to_csv(SAIDA_CSV, index=False, encoding="utf-8-sig")
print(f"\nSalvo: {SAIDA_CSV}")