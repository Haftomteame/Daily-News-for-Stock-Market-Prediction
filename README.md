# Data Lakehouse — Daily News for Stock Market Prediction

Architecture **Medallion** (Bronze / Silver / Gold) pour l'analyse conjointe de données boursières structurées et d'actualités Reddit non structurées.

## Architecture

```
lakehouse/                     ← Source de verite (Parquet)
├── bronze/                    ← Donnees brutes + metadata ingestion
│   ├── stock_prices/          ← OHLCV DJIA (Massive API ou import)
│   ├── news_reddit/           ← Headlines Reddit
│   ├── news_combined/         ← Labels ML + Top1..Top25 (derive)
│   └── massive/day_aggs/      ← Cache flat files Massive (optionnel)
├── silver/                    ← Nettoyage + enrichissement metadata
├── gold/                      ← KPIs journaliers (schema fixe)
└── ml/                        ← Modele + predictions + metrics

monitoring/                    ← Rapports JSON (cout, latence, qualite)

Data/                          ← Legacy CSV (migration ponctuelle uniquement)
```

## Types de donnees

| Type | Couche Bronze | Description |
|------|---------------|-------------|
| **Structure** | `stock_prices` | OHLCV DJIA (Date, Open, High, Low, Close, Volume) |
| **Non structure** | `news_reddit` | Texte libre (headlines Reddit par date) |
| **Hybride** | `news_combined` | News + label ML (0/1) |

## Couches

### BRONZE — ELT (Extract + Load)
- Chargement API (Massive, Arctic Shift) ou lecture bronze existant
- Stockage **Parquet** dans `lakehouse/bronze/`
- Metadata : `_ingestion_ts`, `_source_file`, `_batch_id`, `_layer`
- `news_combined` est **derive** automatiquement depuis stock + reddit

### SILVER — Nettoyage + Metadata
- **Stock** : déduplication, typage, flags qualité (`_quality_score`, `_is_invalid_ohlc`)
- **News** : nettoyage texte, `_word_count`, `_headline_length`, `_has_finance_keyword`

### GOLD — KPIs + Schéma fixe

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | DATE | Date de trading |
| `open/high/low/close` | DOUBLE | Prix DJIA |
| `volume` | BIGINT | Volume échangé |
| `daily_return_pct` | DOUBLE | Rendement journalier (%) |
| `volatility_5d` | DOUBLE | Volatilité glissante 5 jours |
| `news_count` | INTEGER | Nombre d'articles ce jour |
| `avg_headline_length` | DOUBLE | Longueur moyenne des titres |
| `finance_news_ratio` | DOUBLE | Part d'articles finance |
| `market_direction` | VARCHAR | UP / DOWN / FLAT |
| `computed_at` | TIMESTAMP | Horodatage calcul |

## Monitoring

Chaque couche est mesurée sur 3 axes :

| Métrique | Description |
|----------|-------------|
| **Latence** | Temps de traitement (ms) |
| **Coût** | Estimation basée sur le stockage Parquet (tarif S3 ~$0.023/GB) |
| **Qualité** | Score de complétude / validité des données |

Les rapports sont exportés dans `monitoring/report_<batch>.json`.

## Installation

```bash
pip install -r requirements.txt
```

## Stockage : local ou HDFS

Par defaut le lakehouse est sur disque (`lakehouse/`). Pour basculer sur **HDFS** :

```bash
# .env
STORAGE_BACKEND=hdfs
HDFS_NAMENODE=namenode.cluster.local
HDFS_PORT=8020
HDFS_USER=hdfs
HDFS_BASE_PATH=/datax
```

```bash
# 1. Migrer les donnees locales vers HDFS (une fois)
python scripts/upload_local_to_hdfs.py

# 2. Pipeline lit/ecrit directement sur HDFS
python pipeline/run_pipeline.py --massive --predict-year 2026
```

Chemins HDFS :
```
hdfs://namenode:8020/datax/lakehouse/bronze/stock_prices/data.parquet
hdfs://namenode:8020/datax/lakehouse/gold/daily_market_kpis/data.parquet
hdfs://namenode:8020/datax/monitoring/metrics_history.parquet
```

## Execution

```bash
# Migration ponctuelle CSV legacy -> lakehouse (si vous avez deja Data/*.csv)
python scripts/migrate_to_lakehouse.py

# Pipeline complet (lit lakehouse/bronze/ existant)
python pipeline/run_pipeline.py --massive --predict-year 2026

# Re-fetch APIs et re-ecrit bronze
python pipeline/run_pipeline.py --refresh-bronze --massive --predict-year 2026

# Fetch individuel vers lakehouse
python scripts/fetch_massive_rest.py --from 2024-07-08 --to 2026-07-06
python scripts/fetch_reddit_news.py --from 2024-07-08 --to 2026-07-06
python scripts/build_combined_news.py

# Explorer les KPIs Gold (schema fixe)
python scripts/explore_gold.py

# Dashboard interactif (port 8502 par defaut)
python scripts/run_dashboard.py

# Ou manuellement sur un port specifique :
python -m streamlit run dashboard/app.py --server.port 8502

# Tests automatises
pytest tests/ -v
```

Les rapports de monitoring sont exportés dans :
- `monitoring/report_<batch>.json` — snapshot du batch
- `monitoring/metrics_history.parquet` — historique longitudinal (coût, latence, qualité)

## Structure du code

```
src/
├── config.py              ← Chemins, types de donnees, schema Gold
├── bronze/ingest.py       ← Ingestion ELT
├── silver/transform.py    ← Nettoyage + metadata
├── gold/aggregate.py      ← Agregation KPIs (DuckDB)
└── monitoring/metrics.py  ← Metriques par couche

pipeline/run_pipeline.py   ← Orchestrateur (inclut ML)
scripts/explore_gold.py    ← Exploration couche Gold
dashboard/app.py           ← Dashboard Streamlit
tests/                     ← Tests pytest
src/ml/train.py            ← Entrainement ML (Combined_News_DJIA)
```

## ML — Prediction direction marche

Utilise `lakehouse/bronze/news_combined` (Label 0/1) joint aux KPIs Gold :

| Feature | Source |
|---------|--------|
| `daily_return_pct`, `volatility_5d` | Gold |
| `news_count`, `finance_news_ratio` | Gold |
| `combined_finance_ratio`, `combined_avg_length` | Silver (news_combined) |

Modele : **LogisticRegression** — artefacts dans `lakehouse/ml/`.

## Technologies

- **Python 3.10+**
- **Pandas / PyArrow** — manipulation et stockage Parquet
- **DuckDB** — requetes SQL analytiques inter-couches
- **Scikit-learn** — modele ML baseline
- **Streamlit** — dashboard monitoring + KPIs
- **Pytest** — tests automatises
