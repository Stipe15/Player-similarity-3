# ⚽ PlayerMatch

Find who in Europe's top 5 leagues plays like a given player — by **style**, not just position.

Type a name, get a ranked list of statistical look-alikes, and dig into per-player analysis charts: a full stat breakdown, a 2D map of the entire similarity space, and a "how replaceable is this player" score.

Built on 2025/26 season data (1,828 players, 29 per-90 stats, 5 leagues), with a Streamlit interface in Croatian and English.

## How it works

Four "similarity metrics" — cosine, soft-cosine, Euclidean, dot product — turn out to be two knobs on one formula:

```
sim(a, b) = (aᵀ M b) / (aᵀ M a)^(α/2) · (bᵀ M b)^(α/2)
```

- **α** slides between *style* (α=1, ignores volume — a low-minutes player can match a starter) and *volume* (α=0, rewards similar raw output).
- **M** is either the identity matrix or the features' correlation matrix — accounting for stats that naturally move together (e.g. passes and final-third passes) instead of double-counting them.

| Preset | α | M | Answers |
|---|---|---|---|
| Soft-cosine (default) | 1 | correlation | Same style, correlated stats handled sensibly |
| Cosine | 1 | identity | Same style, simplest form |
| Euclidean | 1 | — (distance, not similarity) | Closest overall, style *and* level |
| Dot product | 0 | identity | Same actions in similar *volume* |

Since the dataset has no official position column, player **roles** (e.g. "clearances · blocks", "carries to chance · carries to shot") are instead derived by K-Means clustering the stats themselves — they describe playing style, not a position on the pitch.

## Features

- **Similarity search** with adjustable method, minimum-minutes filter, same-role-only and exclude-same-league filters
- **Player profile pages** — full 29-stat table with percentiles (against all players or just your role), a 2D map of every player in the similarity space (PCA or t-SNE), and a uniqueness score showing whether a player has many close look-alikes or none
- **Two-player comparison** — side-by-side breakdown of what's driving (or breaking) a match
- **Bilingual UI** (Croatian / English), switchable per page
- **Landing page** with a live worked example and FAQ

## Project structure

```
├── app.py              Streamlit app — routing, sidebar, results, player pages, charts
├── similarity.py        Core model: feature space, roles, the unified similarity formula
├── analitika.py         Percentiles, 2D embeddings, uniqueness score, Altair charts
├── prijevodi.py         HR/EN translation dictionary
├── build_dataset.py     Rebuilds podaci/top5_stats_combined.csv from raw per-league exports
├── landing.html          Landing page markup, rendered through app.py
├── projekt.ipynb         Exploratory notebook using the same model as the app
├── podaci/               Data (see below — only the combined CSV is tracked)
├── requirements.txt
└── .streamlit/config.toml
```

`similarity.py` is intentionally the only place the model logic lives — the notebook and the app both import it, so their results can't drift apart.

## Data

Stats are per-90 values across attacking, carrying, defending, and passing, scraped from [theanalyst.com](https://theanalyst.com/) for the 2025/26 half-season across the Premier League, La Liga, Serie A, Bundesliga, and Ligue 1. Goalkeepers are excluded (the feature set is outfield-only).

Only `podaci/top5_stats_combined.csv` is committed — the 20 raw per-league export files are gitignored. To rebuild the combined dataset from scratch:

1. Drop the raw `{league}_{category}.csv` files into `podaci/` (4 categories × 5 leagues).
2. Run:
   ```bash
   python build_dataset.py
   ```

The script **validates that every league is on the same per-90 scale before writing anything** — it will refuse to build if one league's exports turn out to be season totals instead of per-90 values (this caught a real bug during development: Serie A's defensive stats were originally exported as raw totals, which would have made every Serie A player look like a defensive outlier).

## Getting started

Requires Python 3.11+.

```bash
git clone https://github.com/Stipe15/Player-similarity-3.git
cd Player-similarity-3
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

**Run the app:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

**Explore the model in the notebook:**
```bash
jupyter notebook projekt.ipynb
```

## Deploying

The app is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io) for free:

1. Push this repo to GitHub.
2. Sign in at share.streamlit.io with GitHub → **New app**.
3. Select this repo, branch `main`, main file `app.py`.
4. Deploy — `requirements.txt` and `.streamlit/config.toml` are already set up, no extra configuration needed.

## Limitations

- Goalkeepers aren't covered — the stat set is entirely outfield.
- "Roles" are statistical clusters, not official positions; treat them as a style label, not a lineup slot.
- Covers the 2025/26 half-season, not a full season.
