import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))
from paths import get_data_path, get_figure_path, get_result_path, get_report_path

import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.decomposition import PCA

print("Loading cleaned dataset: merged_news_pld_cmo_by_region_date_with_embeddings.csv...")
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_with_embeddings.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

initial_count = len(df)
print(f"Initial total rows: {initial_count}")

# -------------------------------------------------------------
# FILTER OUT 2021 - 2022 TIMELINE (Severe Hydro Drought Crisis)
# -------------------------------------------------------------
mask_exclude_2021_2022 = (df['Date_dt'].dt.year >= 2021) & (df['Date_dt'].dt.year <= 2022)
df_filtered = df[~mask_exclude_2021_2022].copy().reset_index(drop=True)

removed_count = initial_count - len(df_filtered)
print(f"Removed {removed_count} rows from 2021-2022 timeline.")
print(f"Remaining rows for training/testing: {len(df_filtered)} (Years: {sorted(df_filtered['Date_dt'].dt.year.unique())})")

# 1. Columns explicitly dropped as requested:
# Date, Region, submarket_code, day_of_week, month, year, subsystem_id, cmo_reference_date
drop_cols = [
    'Date', 'Date_dt', 'Region', 'submarket_code', 'day_of_week', 
    'month', 'year', 'subsystem_id', 'cmo_reference_date'
]

target_col = 'target_pld_next_day'

# 2. Historical Market Features
hist_market_cols = [
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
    'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
    'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
    'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load'
]

# 3. News Topics & Sentiment Features
news_topic_cols = [
    'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
    'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
    'transmission constraints', 'renewable generation forecasts',
    'demand forecasts (Carga)', 'risk of future energy shortages',
    'Avg Sentiment', 'Avg Importance'
]

def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

print("\nParsing single embedding source: BGE news text embeddings (1024 dimensions)...")
bge_matrix = np.vstack([parse_vector(x, 1024) for x in df_filtered['bg gme embedding']])
bge_cols = [f'bge_dim_{i}' for i in range(bge_matrix.shape[1])]
bge_df = pd.DataFrame(bge_matrix, columns=bge_cols)

# Chronological Train/Test Split (80% Train, 20% Out-of-Time Test)
split_idx = int(len(df_filtered) * 0.8)
y = df_filtered[target_col]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train set: {len(y_train)} rows | Test set: {len(y_test)} rows")

# Apply PCA on BGE embeddings on train set
pca_bge = PCA(n_components=32, random_state=42)
pca_bge.fit(bge_df.iloc[:split_idx])
pca_bge_cols = [f'news_emb_pca_{i+1:02d}' for i in range(32)]

train_pca_bge = pd.DataFrame(pca_bge.transform(bge_df.iloc[:split_idx]), columns=pca_bge_cols, index=df_filtered.index[:split_idx])
test_pca_bge = pd.DataFrame(pca_bge.transform(bge_df.iloc[split_idx:]), columns=pca_bge_cols, index=df_filtered.index[split_idx:])

print(f"BGE Embeddings - 32 PCA Components explain {pca_bge.explained_variance_ratio_.sum()*100:.2f}% of text semantic variance.")

# ==========================================
# EXPERIMENT 1: Baseline (Historical Market Data only)
# ==========================================
X_train_mkt = df_filtered.iloc[:split_idx][hist_market_cols]
X_test_mkt = df_filtered.iloc[split_idx:][hist_market_cols]

rf_mkt = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_mkt.fit(X_train_mkt, y_train)
pred_mkt = rf_mkt.predict(X_test_mkt)

r2_mkt = r2_score(y_test, pred_mkt)
ev_mkt = explained_variance_score(y_test, pred_mkt)
rmse_mkt = np.sqrt(mean_squared_error(y_test, pred_mkt))
mae_mkt = mean_absolute_error(y_test, pred_mkt)

# ==========================================
# EXPERIMENT 2: Market + News Topic & Sentiment Counts
# ==========================================
X_train_topics = df_filtered.iloc[:split_idx][hist_market_cols + news_topic_cols]
X_test_topics = df_filtered.iloc[split_idx:][hist_market_cols + news_topic_cols]

rf_topics = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_topics.fit(X_train_topics, y_train)
pred_topics = rf_topics.predict(X_test_topics)

r2_topics = r2_score(y_test, pred_topics)
ev_topics = explained_variance_score(y_test, pred_topics)
rmse_topics = np.sqrt(mean_squared_error(y_test, pred_topics))
mae_topics = mean_absolute_error(y_test, pred_topics)

# ==========================================
# EXPERIMENT 3: Full Pipeline (Market + Topics + BGE News Embeddings)
# ==========================================
X_train_full = pd.concat([df_filtered.iloc[:split_idx][hist_market_cols + news_topic_cols], train_pca_bge], axis=1)
X_test_full = pd.concat([df_filtered.iloc[split_idx:][hist_market_cols + news_topic_cols], test_pca_bge], axis=1)

rf_full = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_full.fit(X_train_full, y_train)
pred_full = rf_full.predict(X_test_full)

r2_full = r2_score(y_test, pred_full)
ev_full = explained_variance_score(y_test, pred_full)
rmse_full = np.sqrt(mean_squared_error(y_test, pred_full))
mae_full = mean_absolute_error(y_test, pred_full)

# ==========================================
# EXPERIMENT 4: News Embeddings Alone (Variance from text signals alone)
# ==========================================
rf_emb_only = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_emb_only.fit(train_pca_bge, y_train)
pred_emb_only = rf_emb_only.predict(test_pca_bge)

r2_emb_only = r2_score(y_test, pred_emb_only)
ev_emb_only = explained_variance_score(y_test, pred_emb_only)
rmse_emb_only = np.sqrt(mean_squared_error(y_test, pred_emb_only))
mae_emb_only = mean_absolute_error(y_test, pred_emb_only)

# Feature Importance Breakdown
imp_series = pd.Series(rf_full.feature_importances_, index=X_train_full.columns).sort_values(ascending=False)

# Results Dict
results = {
    "Timeline_Filter": "Excluded 2021-2022 (Drought Crisis Anomaly)",
    "Total_Rows": len(df_filtered),
    "Train_Rows": len(y_train),
    "Test_Rows": len(y_test),
    "Baseline_Market_Only": {"R2": round(r2_mkt, 4), "Explained_Variance": round(ev_mkt, 4), "RMSE": round(rmse_mkt, 4), "MAE": round(mae_mkt, 4)},
    "Market_Plus_Topics": {"R2": round(r2_topics, 4), "Explained_Variance": round(ev_topics, 4), "RMSE": round(rmse_topics, 4), "MAE": round(mae_topics, 4)},
    "Full_Pipeline_With_BGE_Embeddings": {"R2": round(r2_full, 4), "Explained_Variance": round(ev_full, 4), "RMSE": round(rmse_full, 4), "MAE": round(mae_full, 4)},
    "News_Embeddings_Alone": {"R2": round(r2_emb_only, 4), "Explained_Variance": round(ev_emb_only, 4), "RMSE": round(rmse_emb_only, 4), "MAE": round(mae_emb_only, 4)},
    "Top_15_Feature_Importances": imp_series.head(15).to_dict()
}

with open(get_result_path("rf_news_embeddings_no_2021_2022_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "="*70)
print("RANDOM FOREST BENCHMARK RESULTS (EXCLUDING 2021-2022)")
print("="*70)
print(f"1. Baseline (Historical Market Data only):")
print(f"   - R2 Score:           {r2_mkt:.4f} ({r2_mkt*100:.2f}%)")
print(f"   - Explained Variance: {ev_mkt:.4f} ({ev_mkt*100:.2f}%)")
print(f"   - RMSE:               {rmse_mkt:.4f}")
print(f"   - MAE:                {mae_mkt:.4f}")

print(f"\n2. Market Features + News Topic Counts:")
print(f"   - R2 Score:           {r2_topics:.4f} ({r2_topics*100:.2f}%)")
print(f"   - Explained Variance: {ev_topics:.4f} ({ev_topics*100:.2f}%)")
print(f"   - RMSE:               {rmse_topics:.4f}")
print(f"   - MAE:                {mae_topics:.4f}")

print(f"\n3. Full Pipeline (Market + Topics + BGE News Embeddings):")
print(f"   - R2 Score:           {r2_full:.4f} ({r2_full*100:.2f}%)")
print(f"   - Explained Variance: {ev_full:.4f} ({ev_full*100:.2f}%)")
print(f"   - RMSE:               {rmse_full:.4f}")
print(f"   - MAE:                {mae_full:.4f}")

print(f"\n4. News Embeddings Alone (Text Signals Only):")
print(f"   - R2 Score:           {r2_emb_only:.4f}")
print(f"   - Explained Variance: {ev_emb_only:.4f}")
print(f"   - RMSE:               {rmse_emb_only:.4f}")
print(f"   - MAE:                {mae_emb_only:.4f}")

print("\nTop 15 Feature Importances in Random Forest Full Model:")
for k, v in imp_series.head(15).items():
    print(f"   - {k:<30}: {v:.6f} ({v*100:.3f}%)")

print("\nBenchmark completed and saved to rf_news_embeddings_no_2021_2022_results.json")
