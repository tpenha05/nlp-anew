# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Coursework for Insper's NLP course (Tiago Tavares — <https://tiagoft.github.io/nlp_course/>), unit *Text Classification*, activity **01 - Practice - Sentiment Analysis with ANEW** — the handcrafted-classifier warm-up that precedes logistic regression and BERT later in the course. By Henrique Mayor, Leonardo Freitas and Thiago Penha.

The assignment asks for a small **"social media monitor"**: pick an entity, download posts about it, decide whether they are on average positive or negative (a `neutral` class is explicitly permitted), and present the result. This team's entity is the film *The Odyssey* (Christopher Nolan, 2026), and the posts are IMDb user reviews.

Two of the assignment's "expected challenges" are where essentially all the design work in this repo lives, and they explain why the code looks the way it does:

- **ANEW gives per-word ratings but no per-post rating** — combining words into one decision is left entirely to the student. `pontuar()` in `scripts.py` *is* that design decision.
- **ANEW has inconsistencies that must be dealt with** — `LIMIAR_SENTIMENTO` (drop near-neutral terms) and `STOPLIST_ENREDO` (drop plot/genre vocabulary) are this project's answers.

The lexicon is the original Bradley & Lang (1999) ANEW, in the CSV form published at <https://osf.io/y6g5b/wiki/anew/> (`pleasure`, `arousal`, `dominance` — the paper's three affective dimensions; note that high arousal does **not** imply negative sentiment).

The central methodological constraint: the corpus deliberately contains **no sentiment column**. The 1–10 `rating` each reviewer gave is kept as raw data and used *only as ground truth to evaluate* the lexicon-derived labels — never as a training label or as an input to scoring. Rating-based evaluation is not required by the assignment; it is this team's addition and the reason `evaluate.py` exists.

Code, comments, and all console output are in Portuguese. `scripts.py` and `evaluate.py` are written without accents on purpose (Windows console encoding); `data/coleta_imdb_odyssey.py` uses full accents. Match the file you are editing.

## Deliverable and constraints

The graded artifact is **not** the code. It is a two-slide presentation exported as PDF and submitted on Blackboard:

1. **How the algorithm works** — rationale, figures and equations allowed, **no source code on the slide**. Written for classmates who solved the same exercise differently: same background, no knowledge of these specific choices.
2. **The results, as a figure** — self-sufficient, showing sentiment toward the entity. The assignment states outright: **never use a pie chart.**

Nothing in the repo currently produces a figure, so the results slide is still an open gap — the scripts only print to the console and write `imdb_odyssey_scores_v2.csv`.

The course's stated AI policy applies directly to work done here: AI should handle menial tasks, not substitute for the students' own reasoning, and **no AI-generated code should be used that they cannot critically review**. Practically, when contributing: prefer short, explicit, readable code over clever abstractions; keep every tunable a named module-level constant with a comment saying why that value; and explain the reasoning behind a change rather than just handing over a diff. The graded learning outcomes are reading documentation and papers autonomously and communicating results clearly — so pointing at the relevant docs or the ANEW paper is more useful than writing the code for them.

## Commands

Dependencies (no requirements file — install manually): `pip install pandas numpy httpx`. **`pandas` is not installed in the default interpreter** (`C:\Users\leosf\AppData\Local\Programs\Python\Python312`), so `scripts.py` and `evaluate.py` do not currently run there; `anew_classificador.py` is stdlib-only and does.

```powershell
# The deliverable: labels every review positivo/negativo, one output CSV. Stdlib only.
python anew_classificador.py

# Score the corpus + print metrics; writes imdb_odyssey_scores_v2.csv to repo root
python scripts.py

# Full evaluation report (5 blocks: metrics, ranges, error terms, worst cases, ablations)
python evaluate.py

# Re-collect the corpus from IMDb — MUST run from inside data/, see gotcha below
cd data; python coleta_imdb_odyssey.py
```

There are no tests, linter, or build. `scripts.py` embeds a hardcoded `TESTES` list of five sentences that it scores and prints before touching the corpus — that list is the de facto smoke test for scoring changes.

### Path gotcha

`scripts.py` and `evaluate.py` use paths relative to the **repo root** (`data/anew.csv`, `data/imdb_odyssey_reviews.csv`), so they must be run from the root. `coleta_imdb_odyssey.py` writes `imdb_odyssey_reviews.csv` and its checkpoint relative to the **cwd**, so it must be run from `data/` or it will drop the CSV in the wrong place.

## The ANEW scale trap

`data/anew.csv` is **not** on the paper's scale. Its values are the Table 1 (all-subjects) SAM ratings multiplied by `100/8.82`, where 8.82 is ANEW's highest `pleasure` (`triumphant`, which becomes exactly 100.0). Verified term by term: the csv/paper ratio is 11.337868 for all three dimensions.

The consequence: the SAM neutral point of 5 ("neither happy nor sad" in the instructions given to raters) sits at **56.69** in this file, not 50. Anything treating 50 as neutral counts every word in [50, 56.69) as pleasant when the paper rates it unpleasant — that band holds a lot of common vocabulary. `scripts.py` has `CENTRO = 50.0` and is affected; `anew_classificador.py` converts back to the 1–9 scale first and centers on 5.

Observed ranges in the file: pleasure 14.17–100.0, arousal 27.10–92.63, dominance 25.74–89.34, 1034 terms, no duplicates, no nulls.

## Architecture

Four stages, each a standalone script; the CSVs in `data/` are the interfaces between them. Stages 2–3 are the team's earlier pandas-based exploration; stage 4 is the self-contained classifier that produces the graded artifact.

**1. `data/coleta_imdb_odyssey.py` — collection.** Pages IMDb's public GraphQL endpoint (`caching.graphql.imdb.com`) instead of scraping HTML. Non-obvious constraints discovered by testing and documented in the module docstring: HTML scraping returns `HTTP 202` with an empty body for non-browser clients; `Origin`/`Referer` headers are mandatory (403 without them); `first` caps at 50 per page; sorting by `SUBMISSION_DATE` (not "most helpful") is required for deterministic pagination. Checkpoints every 10 pages to `imdb_odyssey_checkpoint.json` and resumes from it; deletes the checkpoint on success. Ends with a quality report that sanity-checks the collected rating mean against IMDb's public aggregate.

**2. `scripts.py` — the scorer.** This is the model. Its entire behavior is controlled by module-level constants at the top of the file, which is what makes stage 3 possible.

- `carregar_lexico` reads `data/anew.csv` (1034 ANEW terms; columns `term`, `pleasure`, `arousal`, `dominance` on a 0–100 scale), normalizes valence/arousal to −1..1 around 50, and **drops** terms below `LIMIAR_SENTIMENTO` (too neutral to carry opinion) and terms in `STOPLIST_ENREDO`.
- `STOPLIST_ENREDO` exists because Odyssey plot vocabulary (`war`, `sea`, `god`, `hero`, `death`…) and generic film words carry strong ANEW valence but zero opinion about the film. `termos_dominantes()` in `scripts.py` and block 4 of `evaluate.py` exist to find new candidates for it.
- `LEXICO_EXTRA` is hand-authored film-criticism vocabulary (`masterpiece`, `tedious`, `overrated`…) ANEW does not cover. It is gated behind `USAR_LEXICO_EXTRA = False` because enabling it means the experiment is **no longer pure ANEW** — a methodological choice, not a performance one. Do not flip it silently.
- `pontuar` tokenizes, applies negation (window of `JANELA_NEGACAO` words, flips valence at 0.8 strength), intensifier multipliers, and counts the review title `PESO_TITULO` times. The score is a polarity index: `(positive_mass − negative_mass) / total_mass`, so it is bounded in −1..1 and independent of review length.
- `rotular` maps score → `positivo`/`negativo`/`neutro`, or `indefinido` when fewer than `MIN_TERMOS` lexicon terms matched. Ground truth mapping is `rating >= 7` → positive, `<= 4` → negative, else neutral (defined in both files — keep them in sync).

**3. `evaluate.py` — the harness.** `import scripts as base` and mutates `base`'s constants with `setattr` to run ablations (`bloco_ablacoes` toggles stoplist, extra lexicon, negation, title weight, valence/arousal weights, sentiment threshold, then restores the defaults). Any new tunable in `scripts.py` should be a module-level constant so it can be ablated this way.

The key evaluation idea: `melhores_limiares` grid-searches the label cutoffs to maximize macro F1, which separates *quality of the score* (Spearman vs. rating) from *quality of the cutoffs* (F1 macro). High Spearman + low F1 means only calibration is missing; the reverse means the cutoffs got lucky. Accuracy is always compared against the majority-class baseline. `scripts.calibrar()` suggests `LIMIAR_POS`/`LIMIAR_NEG` from the rating distribution's quantiles.

`evaluate.py` depends on `n_palavras` from the collector's output, so a hand-made corpus CSV needs that column for block 3.

**4. `anew_classificador.py` — the classifier that produces the deliverable.** Reads `data/anew.csv` + `data/imdb_odyssey_reviews.csv`, writes exactly one output, `imdb_odyssey_reviews_anew.csv`: every original column preserved byte-for-byte plus `anew_score`, `anew_termos`, and `sentimento` (strictly `positivo`/`negativo` for all 4498 rows). Setting `COLUNAS_DIAGNOSTICO = False` reduces that to just the `sentimento` column.

Independent of `scripts.py` by design — **stdlib only, no pandas/numpy** — so it runs on the default interpreter, and short enough to be critically reviewed line by line as the course's AI policy requires. Its choices are grounded in the manual rather than in tuning: `pleasure` supplies polarity, `arousal` only scales magnitude and can never flip a sign (the paper treats arousal as calm↔excited, orthogonal to valence), `dominance` is unused, and the decision threshold defaults to 0.0 — the SAM neutral point — rather than a value fitted to the ratings.

Measured on the corpus (balanced accuracy at the best possible threshold, 65.6% for the shipped config):

| Component | Effect if removed |
|---|---|
| `STOPLIST_ENREDO` | 65.6% → 63.8%; at threshold 0, 62.6% → 56.0% — the single biggest contributor |
| `USAR_LEMATIZACAO_SIMPLES` | 65.6% → 61.7% |
| negation window | 65.6% → 63.2% |
| `PESO_TITULO` (2 → 0) | 65.6% → 63.2% |
| `W_AROUSAL` (0.25 → 0) | 65.6% → 65.5% — no measurable effect |

The honest headline: at the paper's neutral threshold the classifier gets 75.4% accuracy against a 77.2% majority baseline, i.e. **raw accuracy does not beat guessing "positive" every time**, though balanced accuracy is 62.6% and mean score rises monotonically with rating (+0.057 at rating 1 → +0.542 at rating 10). The cause is that reviewers use pleasant words even when panning the film. Calibrating the threshold to +0.37 lifts balanced accuracy to 65.6% but means the cut was fitted to the ground truth, which must be stated if used. Coverage is 3.1% of tokens (6.2 ANEW words per 202-word review), and 279 reviews match no ANEW word at all and fall to `positivo` via the `>=` tie-break.

## Known divergences from the brief

Deliberate or drifted-into choices worth knowing before "fixing" them:

- **IMDb, not a social media outlet.** The assignment says social media and suggests `requests`/`beautifulsoup`/`selenium`. This project uses IMDb's public GraphQL API with `httpx` — better structured data, but IMDb reviews are longer and more considered than social posts, and they come with a rating, which is what made the extra evaluation possible.
- **`dominance` is unused.** It is read from `data/anew.csv` and dropped; only `pleasure` and `arousal` feed the score (`W_VALENCIA`/`W_AROUSAL`). The assignment expects all three to be understood, not necessarily all used.
- **`PLAN.md`** (the original research plan — deleted in the working tree, still at `HEAD` in git) targets Reddit comments and all three dimensions. It also flags that ANEW's 1034 terms will give low coverage and suggests re-running against the Warriner et al. extension (13,915 lemmas) for comparison. Not done; `evaluate.py`'s coverage metric is the evidence for that concern.
- **Entity choice.** The brief advises against controversial entities; a divisive film release is borderline, and the negative class being the hard one (`f1_negativo` is called out separately in `evaluate.py`) is partly a symptom of that.
