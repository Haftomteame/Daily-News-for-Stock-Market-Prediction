# Data Lakehouse — Daily News for Stock Market Prediction

Architecture **Medallion** (Bronze / Silver / Gold) pour l'analyse conjointe de données boursières structurées et d'actualités Reddit non structurées.

## Architecture

```
lakehouse/                     ← Source de verite (Parquet)
├── bronze/                    ← Donnees brutes + metadata ingestion
│   ├── stock_prices/          ← OHLCV DJIA journalier (Massive API ou import)
│   ├── stock_prices_1m/       ← OHLCV 1 min temps reel (Finnhub WebSocket)
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
| **Structure** | `stock_prices` | OHLCV DJIA journalier (Date, Open, High, Low, Close, Volume) |
| **Structure** | `stock_prices_1m` | OHLCV 1 min temps reel Finnhub (DIA) |
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

## Base de donnees (PostgreSQL) — donnees structurees

Les donnees **structurees** (Silver/Gold) sont aussi chargees dans PostgreSQL (mode **replace** a chaque run) :

- `silver_stock_prices`
- `silver_news_reddit`
- `silver_news_combined`
- `gold_daily_market_kpis`

### Connexion (Docker)

- **Host** : `localhost`
- **Port** : `5432`
- **Database** : `wherehouse`
- **User** : `datax`
- **Password** : `10102026Ha`

Vous pouvez vous connecter avec **DBeaver / pgAdmin / PowerBI**.

## Docker + HDFS (voir le cluster)

Le code HDFS existait mais **aucun service Hadoop** n'etait demarre. Utilisez le profil `hdfs` :

```bash
docker compose --profile hdfs up -d --build
```

Services demarres :
| Service | URL / role |
|---------|------------|
| **NameNode UI** | http://localhost:9870 — explorer les fichiers HDFS |
| **Dashboard HDFS** | http://localhost:8503 — lit les donnees depuis HDFS |
| **PostgreSQL** | `localhost:5432` — base `wherehouse` (tables Silver/Gold) |
| `hdfs-init` | copie `lakehouse/` local vers `/datax` sur HDFS (une fois) |

Dans l'UI HDFS : **Utilities → Browse the file system** → `/datax/lakehouse/`

```bash
# Pipeline sur HDFS
docker compose --profile hdfs --profile pipeline up --build --abort-on-container-exit pipeline-hdfs

# Arreter
docker compose --profile hdfs down
```

Mode local (sans HDFS) : `docker compose up dashboard` → http://localhost:8502

## Streaming temps reel (Finnhub — DIA 1 min)

Flux WebSocket Finnhub → bougies OHLCV 1 minute → `lakehouse/bronze/stock_prices_1m/`.

### Prerequis

1. Cle API dans `.env` :
```env
FINNHUB_TOKEN=votre_cle
FINNHUB_SYMBOL=DIA
```

2. Marche US ouvert (les bougies n'apparaissent que lorsqu'il y a des trades).

### Lancer en local

```bash
python scripts/stream_finnhub_ohlcv.py --ticker DIA --lakehouse --csv
```

- **Lakehouse** : `lakehouse/bronze/stock_prices_1m/data.parquet` (mis a jour en continu)
- **CSV** (option `--csv`) : `Data/finnhub_dia_1m.csv`

### Lancer avec Docker (service long-running)

```bash
# Local
docker compose --profile stream up -d --build finnhub-stream

# HDFS
docker compose --profile hdfs --profile stream up -d --build finnhub-stream-hdfs

# Logs
docker compose logs -f finnhub-stream

# Arreter
docker compose --profile stream down
```

Le dashboard affiche les dernieres bougies 1 min dans l'onglet **Temps reel** (rafraichir pour mettre a jour).

**Note** : le plan gratuit Finnhub peut imposer un delai (~15 min) sur les donnees US.

## Spark (Option A) — Spark pour Gold (ETL), ML reste en scikit-learn

Spark tourne en Docker (profil `spark`) et lit/ecrit directement sur **HDFS**.

### Demarrer Spark

```bash
docker compose --profile hdfs --profile spark up -d spark-master spark-worker
```

- **Spark UI** : http://localhost:8080

### Executer le job Gold avec Spark

```bash
docker compose --profile hdfs --profile spark run --rm spark-gold
```

Le job lit :
- `hdfs://namenode:8020/datax/lakehouse/silver/stock_prices/data.parquet`
- `hdfs://namenode:8020/datax/lakehouse/silver/news_reddit/data.parquet`

et ecrit :
- `hdfs://namenode:8020/datax/lakehouse/gold/daily_market_kpis/data.parquet` (overwrite)

## Airflow — Orchestration du pipeline

Le DAG `stock_market_lakehouse` enchaine les etapes du lakehouse avec monitoring et chargement PostgreSQL.

```
init_batch → wait_hdfs → bronze_ingest → silver_transform → gold_spark
  → ml_train → postgres_warehouse → monitoring_report
```

### Demarrer Airflow (HDFS + Spark + Airflow)

```bash
docker compose --profile hdfs --profile spark --profile airflow up -d --build
```

| Service | URL / role |
|---------|------------|
| **Airflow UI** | http://localhost:8081 — login `admin` / `admin` |
| **DAG** | `stock_market_lakehouse` (planifie `@daily`) |
| **Scheduler** | Execute les taches Python sur HDFS |
| **Gold Spark** | Lance `spark-gold` via socket Docker |

### Declencher manuellement

Dans l'UI Airflow : **DAGs** → `stock_market_lakehouse` → **Trigger DAG**.

### Variables utiles (docker-compose)

| Variable | Defaut | Description |
|----------|--------|-------------|
| `GOLD_ENGINE` | `spark` | `spark` ou `python` (DuckDB) |
| `PIPELINE_MASSIVE` | `true` | Source Massive pour stock_prices |
| `PIPELINE_PREDICT_YEAR` | `2026` | Annee de prediction ML |
| `AIRFLOW_DAG_SCHEDULE` | `@daily` | Cron Airflow |

### CLI alternative (sans Airflow)

```bash
python scripts/pipeline_task.py bronze --batch-id <uuid>
python scripts/pipeline_task.py gold_spark --batch-id <uuid>
```

## Stockage : local ou HDFS

Par defaut le lakehouse est sur disque (`lakehouse/`). Pour basculer sur **HDFS** :

```bash
# .env
STORAGE_BACKEND=hdfs
HDFS_NAMENODE=namenode
HDFS_WEB_PORT=9870
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

# Gold via Spark (HDFS)
python pipeline/run_pipeline.py --massive --predict-year 2026 --gold-engine spark

# Re-fetch APIs et re-ecrit bronze
python pipeline/run_pipeline.py --refresh-bronze --massive --predict-year 2026

# Fetch individuel vers lakehouse
python scripts/fetch_massive_rest.py --from 2024-07-08 --to 2026-07-06

# Stream temps reel Finnhub (DIA, bougies 1 min -> lakehouse/bronze/stock_prices_1m/)
# FINNHUB_TOKEN dans .env ; marche US ouverte recommande
python scripts/stream_finnhub_ohlcv.py --ticker DIA --lakehouse --csv

# Ou via Docker (service long-running)
docker compose --profile stream up -d finnhub-stream
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
├── db/postgres.py          ← Ecriture/lecture PostgreSQL (warehouse)
├── pipeline/lakehouse_tasks.py  ← Taches modulaires (CLI + Airflow)
├── bronze/ingest.py       ← Ingestion ELT
├── silver/transform.py    ← Nettoyage + metadata
├── gold/aggregate.py      ← Agregation KPIs (DuckDB)
└── monitoring/metrics.py  ← Metriques par couche

dags/stock_market_lakehouse_dag.py  ← DAG Airflow
pipeline/run_pipeline.py   ← Orchestrateur monolithique (inclut ML)
scripts/pipeline_task.py   ← Une tache pipeline en CLI
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
- **Apache Airflow** — orchestration DAG (profil `airflow`)
- **Pytest** — tests automatises
