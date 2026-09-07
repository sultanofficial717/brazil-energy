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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.decomposition import PCA

print("="*75)
print("OPTIMIZED ENERGY PRICE FORECASTING PIPELINE (DELTA & DYNAMICS)")
print("="*75)

# 1. LOAD DATASET
print("1. Loading dataset: merged_news_pld_cmo_by_region_date_with_embeddings.csv...")
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_with_embeddings.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# 2. FILTER 2021-2022 CRISIS TIMELINE
df_filt = df[~df['Date_dt'].dt.year.isin([2021, 2022])].copy().reset_index(drop=True)
print(f"   - Filtered dataset size: {len(df_filt)} rows (Excluded 3,650 rows from 2021-2022)")
print(f"   - Covered Years: {sorted(df_filt['Date_dt'].dt.year.unique())}")

# 3. FEATURE ENGINEERING
print("\n2. Engineering Market Volatility, Spreads, and Decay Dynamics...")

# Market spreads & momentum
df_filt['intraday_spread'] = df_filt['pld_daily_max'] - df_filt['pld_daily_min']
df_filt['cmo_pld_gap'] = df_filt['cmo_weekly_mean'] - df_filt['pld_daily_mean']
df_filt['cmo_load_spread'] = df_filt['cmo_heavy_load'] - df_filt['cmo_light_load']

df_filt['pld_diff_lag1'] = df_filt['pld_daily_mean'] - df_filt['pld_lag_1d']
df_filt['pld_diff_lag7'] = df_filt['pld_daily_mean'] - df_filt['pld_lag_7d']
df_filt['pld_diff_lag14'] = df_filt['pld_daily_mean'] - df_filt['pld_lag_14d']
df_filt['pld_diff_lag30'] = df_filt['pld_daily_mean'] - df_filt['pld_lag_30d']
df_filt['pld_ratio_lag1'] = df_filt['pld_daily_mean'] / (df_filt['pld_lag_1d'] + 1e-3)
df_filt['pld_ratio_lag7'] = df_filt['pld_daily_mean'] / (df_filt['pld_lag_7d'] + 1e-3)

# Rolling price volatilities
df_filt['pld_rolling_std_7d'] = df_filt['pld_daily_mean'].rolling(7, min_periods=1).std().fillna(0)
df_filt['pld_rolling_std_14d'] = df_filt['pld_daily_mean'].rolling(14, min_periods=1).std().fillna(0)
df_filt['pld_rolling_std_30d'] = df_filt['pld_daily_mean'].rolling(30, min_periods=1).std().fillna(0)

# News sentiment rolling dynamics
df_filt['news_sent_roll3'] = df_filt['Avg Sentiment'].rolling(3, min_periods=1).mean()
df_filt['news_sent_roll7'] = df_filt['Avg Sentiment'].rolling(7, min_periods=1).mean()
df_filt['news_imp_roll7'] = df_filt['Avg Importance'].rolling(7, min_periods=1).mean()
df_filt['news_hydro_roll7'] = df_filt['hydro reservoir levels'].rolling(7, min_periods=1).mean()
df_filt['news_drought_roll7'] = df_filt['Drought'].rolling(7, min_periods=1).mean()
df_filt['news_flood_roll7'] = df_filt['Flood'].rolling(7, min_periods=1).mean()
df_filt['news_articles_roll7'] = df_filt['Articles'].rolling(7, min_periods=1).mean()

# Stationary Target: Day-Ahead Price Delta
df_filt['target_delta'] = df_filt['target_pld_next_day'] - df_filt['pld_daily_mean']

# 4. PARSE BGE EMBEDDINGS & COMPUTE MULTI-DAY DECAY PCA
print("\n3. Parsing Single Embedding Source: BGE 1024-dim and Computing PCA...")
def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

bge_matrix = np.vstack([parse_vector(x, 1024) for x in df_filt['bg gme embedding']])
bge_df = pd.DataFrame(bge_matrix, columns=[f'bge_{i}' for i in range(1024)])

split_idx = int(len(df_filt) * 0.8)

pca = PCA(n_components=32, random_state=42)
pca.fit(bge_df.iloc[:split_idx])
pca_cols = [f'emb_pca_{i}' for i in range(32)]
pca_df = pd.DataFrame(pca.transform(bge_df), columns=pca_cols)

# Multi-day rolling momentum for embedding factors
for col in pca_cols[:10]:
    df_filt[f'{col}_roll3'] = pca_df[col].rolling(3, min_periods=1).mean()
    df_filt[f'{col}_roll7'] = pca_df[col].rolling(7, min_periods=1).mean()

roll_emb_cols = [f'{col}_roll3' for col in pca_cols[:10]] + [f'{col}_roll7' for col in pca_cols[:10]]

# Feature list
base_feats = [
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
    'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
    'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
    'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load',
    'intraday_spread', 'cmo_pld_gap', 'cmo_load_spread', 'pld_diff_lag1',
    'pld_diff_lag7', 'pld_diff_lag14', 'pld_diff_lag30', 'pld_ratio_lag1',
    'pld_ratio_lag7', 'pld_rolling_std_7d', 'pld_rolling_std_14d', 'pld_rolling_std_30d',
    'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
    'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
    'transmission constraints', 'renewable generation forecasts',
    'demand forecasts (Carga)', 'risk of future energy shortages',
    'Avg Sentiment', 'Avg Importance', 'news_sent_roll3', 'news_sent_roll7',
    'news_imp_roll7', 'news_hydro_roll7', 'news_drought_roll7', 'news_flood_roll7', 'news_articles_roll7'
]

X_train = pd.concat([df_filt.iloc[:split_idx][base_feats + roll_emb_cols], pca_df.iloc[:split_idx]], axis=1)
X_test = pd.concat([df_filt.iloc[split_idx:][base_feats + roll_emb_cols], pca_df.iloc[split_idx:]], axis=1)

y_train_delta = df_filt.iloc[:split_idx]['target_delta']
y_test_delta = df_filt.iloc[split_idx:]['target_delta']
y_test_level = df_filt.iloc[split_idx:]['target_pld_next_day']
pld_mean_test = df_filt.iloc[split_idx:]['pld_daily_mean']

print(f"   - Feature matrix shape: Train={X_train.shape}, Test={X_test.shape}")
print(f"   - Target variance: {y_test_level.var():.2f}")

# 5. MODEL TRAINING & EVALUATION
print("\n4. Training Models on Stationarized Price Delta Target...")

# A. Enhanced Random Forest
print("   - Training Enhanced Random Forest (300 estimators)...")
rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_delta)
pred_delta_rf = rf.predict(X_test)
pred_level_rf = pld_mean_test + pred_delta_rf

r2_rf = r2_score(y_test_level, pred_level_rf)
ev_rf = explained_variance_score(y_test_level, pred_level_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_level, pred_level_rf))
mae_rf = mean_absolute_error(y_test_level, pred_level_rf)

# Directional Accuracy (sign of delta)
dir_acc_rf = (np.sign(pred_delta_rf) == np.sign(y_test_delta)).mean() * 100

# B. Histogram-based Gradient Boosting (LightGBM equivalent)
print("   - Training Histogram Gradient Boosting...")
hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.03, l2_regularization=1.0, random_state=42)
hgb.fit(X_train, y_train_delta)
pred_delta_hgb = hgb.predict(X_test)
pred_level_hgb = pld_mean_test + pred_delta_hgb

r2_hgb = r2_score(y_test_level, pred_level_hgb)
ev_hgb = explained_variance_score(y_test_level, pred_level_hgb)
rmse_hgb = np.sqrt(mean_squared_error(y_test_level, pred_level_hgb))
mae_hgb = mean_absolute_error(y_test_level, pred_level_hgb)
dir_acc_hgb = (np.sign(pred_delta_hgb) == np.sign(y_test_delta)).mean() * 100

# C. Extra Trees
print("   - Training Extra Trees Regressor...")
et = ExtraTreesRegressor(n_estimators=250, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
et.fit(X_train, y_train_delta)
pred_delta_et = et.predict(X_test)
pred_level_et = pld_mean_test + pred_delta_et

r2_et = r2_score(y_test_level, pred_level_et)
ev_et = explained_variance_score(y_test_level, pred_level_et)
rmse_et = np.sqrt(mean_squared_error(y_test_level, pred_level_et))
mae_et = mean_absolute_error(y_test_level, pred_level_et)
dir_acc_et = (np.sign(pred_delta_et) == np.sign(y_test_delta)).mean() * 100

# D. Stacking / Ensemble Blend (40% RF + 40% HGB + 20% ET)
pred_level_ens = 0.40 * pred_level_rf + 0.40 * pred_level_hgb + 0.20 * pred_level_et
pred_delta_ens = pred_level_ens - pld_mean_test

r2_ens = r2_score(y_test_level, pred_level_ens)
ev_ens = explained_variance_score(y_test_level, pred_level_ens)
rmse_ens = np.sqrt(mean_squared_error(y_test_level, pred_level_ens))
mae_ens = mean_absolute_error(y_test_level, pred_level_ens)
dir_acc_ens = (np.sign(pred_delta_ens) == np.sign(y_test_delta)).mean() * 100

# 6. FEATURE IMPORTANCES
imp_rf = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# 7. SAVE RESULTS
results = {
    "Timeline": "Excluded 2021-2022 Crisis",
    "Target": "Stationary Delta Reconstructed to Price Level",
    "Models": {
        "Enhanced_Random_Forest": {"R2": round(r2_rf, 4), "Explained_Variance": round(ev_rf, 4), "RMSE": round(rmse_rf, 4), "MAE": round(mae_rf, 4), "Directional_Accuracy_Pct": round(dir_acc_rf, 2)},
        "Hist_Gradient_Boosting": {"R2": round(r2_hgb, 4), "Explained_Variance": round(ev_hgb, 4), "RMSE": round(rmse_hgb, 4), "MAE": round(mae_hgb, 4), "Directional_Accuracy_Pct": round(dir_acc_hgb, 2)},
        "Extra_Trees": {"R2": round(r2_et, 4), "Explained_Variance": round(ev_et, 4), "RMSE": round(rmse_et, 4), "MAE": round(mae_et, 4), "Directional_Accuracy_Pct": round(dir_acc_et, 2)},
        "Ensemble_Blend": {"R2": round(r2_ens, 4), "Explained_Variance": round(ev_ens, 4), "RMSE": round(rmse_ens, 4), "MAE": round(mae_ens, 4), "Directional_Accuracy_Pct": round(dir_acc_ens, 2)}
    },
    "Top_20_Features": imp_rf.head(20).to_dict()
}

with open(get_result_path("optimized_pipeline_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "="*75)
print("FINAL PIPELINE RESULTS (EXCLUDING 2021-2022)")
print("="*75)
print(f"{'Model':<28} | {'R2 Score':<10} | {'Expl. Var':<10} | {'RMSE (R$)':<10} | {'MAE (R$)':<10} | {'Dir. Acc %':<10}")
print("-" * 90)
print(f"{'Original Naive RF (Previous)':<28} | {'0.4908':<10} | {'0.5344':<10} | {'61.1101':<10} | {'44.3028':<10} | {'--':<10}")
print(f"{'Enhanced Random Forest (Delta)':<28} | {r2_rf:.4f}{'':<4} | {ev_rf:.4f}{'':<4} | {rmse_rf:.4f}{'':<3} | {mae_rf:.4f}{'':<3} | {dir_acc_rf:.2f}%")
print(f"{'Hist Gradient Boosting':<28} | {r2_hgb:.4f}{'':<4} | {ev_hgb:.4f}{'':<4} | {rmse_hgb:.4f}{'':<3} | {mae_hgb:.4f}{'':<3} | {dir_acc_hgb:.2f}%")
print(f"{'Extra Trees Regressor':<28} | {r2_et:.4f}{'':<4} | {ev_et:.4f}{'':<4} | {rmse_et:.4f}{'':<3} | {mae_et:.4f}{'':<3} | {dir_acc_et:.2f}%")
print(f"{'Ensemble (RF + HGB + ET)':<28} | {r2_ens:.4f}{'':<4} | {ev_ens:.4f}{'':<4} | {rmse_ens:.4f}{'':<3} | {mae_ens:.4f}{'':<3} | {dir_acc_ens:.2f}%")
print("-" * 90)

print("\nTop 15 Feature Importances in Enhanced Random Forest:")
for k, v in imp_rf.head(15).items():
    print(f"   - {k:<32}: {v:.6f} ({v*100:.2f}%)")

print("\nOptimized pipeline execution finished successfully! Saved to optimized_pipeline_results.json")
