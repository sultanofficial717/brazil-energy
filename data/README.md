# Dataset Documentation & Data Dictionary

This directory contains the market, hydrological, and NLP-embedded news datasets utilized for forecasting Brazilian wholesale electricity prices (**PLD** - *Preço de Liquidação das Diferenças*) and the system marginal operating cost (**CMO** - *Custo Marginal de Operação*).

---

## 1. Primary Datasets

### A. `daily_train_ready.csv` (~16.5 MB)
- **Rows**: 11,852 daily observations
- **Columns**: 137 engineered features
- **Included in Git**: Yes (within GitHub size limit)
- **Description**: Fully preprocessed daily time-series ready for machine learning (XGBoost, Random Forest, HistGradientBoosting, LSTM). Includes:
  - Historical price lags: `pld_lag_1d`, `pld_lag_7d`, `pld_lag_14d`, `pld_lag_30d`
  - Intraday volatility & spreads: `intraday_spread`, `cmo_pld_gap`, `cmo_load_spread`
  - Multi-window rolling statistics: 7-day, 14-day, and 30-day means and standard deviations
  - Dimensionally reduced news representations: PCA components (`news_reg_embedding_pca_01` through `pca_10`) and rolling momentum metrics.

### B. `merged_news_pld_cmo_by_region_date_with_embeddings.csv` (~720 MB)
- **Rows**: 16,178 observations across Brazilian submarkets (SE/CO, S, NE, N)
- **Columns**: 41 columns including the 1024-dimensional dense embedding vector `bg gme embedding`
- **Large File Note**: Excluded from default Git tracking (exceeds GitHub's 100 MB file limit). See [Storage & Download Instructions](#storage--large-file-instructions) below.

### C. `merged_news_pld_cmo_by_region_date_clean.csv` (~736 MB)
- Cleaned and aligned version of the merged dataset with temporal ordering and verified target values.

### D. Benchmark Test Predictions
- `predictions_12h_with_news.csv` (~325 MB) - Out-of-time test predictions at the 12-hour sub-daily horizon.
- `predictions_24h_with_news.csv` (~162 MB) - Out-of-time test predictions at the 24-hour horizon.
- `test_preds_12h_official.csv` (~325 MB) - Official test set predictions across Market-Only, News-Alone, and Full Pipeline models (N=3,262).
- `test_preds_3d_official.csv` (~311 KB) - Official test set predictions at the 3-day window (N=547).

---

## 2. Key Variables & Features

| Column Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `Date` / `Date_dt` | Date | YYYY-MM-DD | Timestamp of the trading / dispatch day |
| `Region` | String | Submarket | Brazilian electric submarket (`Southeast`, `South`, `Northeast`, `North`) |
| `pld_daily_mean` | Float | R\$/MWh | Daily average spot clearing price (PLD) |
| `pld_daily_min` / `max` | Float | R\$/MWh | Daily price floor and ceiling in the submarket |
| `cmo_weekly_mean` | Float | R\$/MWh | Operative marginal cost determined by ONS optimization models (DECOMP/NEWAVE) |
| `Target_Gap` | Float | R\$/MWh | Basis dislocation spread: $\text{CMO} - \text{PLD}$ |
| `target_delta` | Float | R\$/MWh | Price change relative to baseline: $PLD_{t+h} - PLD_{t}$ |
| `Articles` | Integer | $\ge 0$ | Total news articles published on that date mentioning energy/hydrology |
| `Drought` | Integer | $\ge 0$ | Count of keyword mentions related to drought, low precipitation, and water stress |
| `Flood` | Integer | $\ge 0$ | Count of keyword mentions related to floods, heavy rainfall, and spillway openings |
| `hydro reservoir levels` | Integer | $\ge 0$ | Mentions regarding Ear (Stored Energy) and reservoir percentages |
| `thermal fuel costs` | Integer | $\ge 0$ | Mentions regarding LNG, diesel, and thermal dispatch activation costs |
| `Curtailment` | Integer | $\ge 0$ | Mentions regarding solar/wind generation curtailment and transmission bottlenecks |
| `Avg Sentiment` | Float | [-1.0, +1.0] | Average lexical sentiment score of published articles |
| `bg gme embedding` | JSON String | $\mathbb{R}^{1024}$ | 1024-dimensional dense semantic embedding vector extracted using BAAI/BGE models |

---

## 3. Storage & Large File Instructions

GitHub imposes a strict **100 MB** limit per file. Datasets larger than 100 MB (`merged_news_pld_cmo_by_region_date_*.csv`, `predictions_*.csv`) are ignored by `.gitignore` to maintain a lightweight repository.

To obtain or re-create the raw embeddings dataset:
1. Ensure the raw article feeds and CCEE/ONS market data are placed into `data/`.
2. Run `scripts/train_optimized_pipeline.py` or `scripts/train_windows_benchmark.py` to regenerate predictions and benchmarks locally.
3. For cloud storage links (HuggingFace Hub / Zenodo / Google Drive), configure your remote storage URI in `paths.py`.
