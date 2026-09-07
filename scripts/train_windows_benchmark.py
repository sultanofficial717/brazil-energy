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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

print("="*90)
print("MULTI-GRANULARITY BENCHMARK: 3-DAY TIME WINDOW & 12-HOUR TIME WINDOW")
print("="*90)

# 1. LOAD RAW MERGED DATASET
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_clean.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# Parse BGE Embeddings
def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

print("Parsing 1024-dim BGE news embeddings...")
bge_matrix = np.vstack([parse_vector(x, 1024) for x in df['bg gme embedding']])
bge_cols = [f'bge_{i}' for i in range(1024)]
bge_df = pd.DataFrame(bge_matrix, columns=bge_cols)

# Retain only news-active rows (Articles >= 1)
news_mask = (df['Articles'] > 0).values
df_clean = df[news_mask].copy().reset_index(drop=True)
bge_clean = bge_df[news_mask].copy().reset_index(drop=True)

# -------------------------------------------------------------
# FUNCTION: TRAIN & EVALUATE ENSEMBLE PIPELINE
# -------------------------------------------------------------
def train_and_eval(X_train, X_test, y_train_delta, y_test_delta, y_test_level, base_level_test):
    m_rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    m_hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.03, l2_regularization=1.0, random_state=42)
    m_et = ExtraTreesRegressor(n_estimators=250, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    
    m_rf.fit(X_train, y_train_delta)
    m_hgb.fit(X_train, y_train_delta)
    m_et.fit(X_train, y_train_delta)
    
    pred_delta = 0.40 * m_rf.predict(X_test) + 0.40 * m_hgb.predict(X_test) + 0.20 * m_et.predict(X_test)
    pred_level = base_level_test + pred_delta
    
    r2 = r2_score(y_test_level, pred_level)
    ev = explained_variance_score(y_test_level, pred_level)
    rmse = np.sqrt(mean_squared_error(y_test_level, pred_level))
    mae = mean_absolute_error(y_test_level, pred_level)
    dir_acc = (np.sign(pred_delta) == np.sign(y_test_delta)).mean() * 100
    
    return {
        "R2": round(r2, 4),
        "Explained_Variance": round(ev, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Directional_Accuracy_Pct": round(dir_acc, 2),
        "RF_Model": m_rf
    }

# =============================================================
# PART 1: 3-DAY TIME WINDOW PIPELINE (aggregate_3d)
# =============================================================
print("\n" + "-"*60)
print("PROCESSING 3-DAY TIME WINDOW (Multi-Day Aggregated Blocks)")
print("-"*60)

df_3d = df_clean.copy()
df_3d['_group_3d'] = np.arange(len(df_3d)) // 3

# Numeric features to average across 3-day blocks
num_cols = [
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
    'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
    'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
    'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load',
    'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
    'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
    'transmission constraints', 'renewable generation forecasts',
    'demand forecasts (Carga)', 'risk of future energy shortages',
    'Avg Sentiment', 'Avg Importance'
]

# Aggregate market data over 3 days
agg_dict = {c: 'mean' for c in num_cols}
agg_dict['target_pld_next_day'] = 'mean'
agg_3d_df = df_3d.groupby('_group_3d').agg(agg_dict).reset_index(drop=True)

# Aggregate embeddings over 3 days
bge_3d = bge_clean.groupby(df_3d['_group_3d']).agg('mean').reset_index(drop=True)

# 3-Day Spreads & Dynamics
agg_3d_df['intraday_spread'] = agg_3d_df['pld_daily_max'] - agg_3d_df['pld_daily_min']
agg_3d_df['cmo_pld_gap'] = agg_3d_df['cmo_weekly_mean'] - agg_3d_df['pld_daily_mean']
agg_3d_df['pld_diff_lag1'] = agg_3d_df['pld_daily_mean'] - agg_3d_df['pld_lag_1d']
agg_3d_df['pld_diff_lag7'] = agg_3d_df['pld_daily_mean'] - agg_3d_df['pld_lag_7d']
agg_3d_df['pld_ratio_lag1'] = agg_3d_df['pld_daily_mean'] / (agg_3d_df['pld_lag_1d'] + 1e-3)
agg_3d_df['pld_rolling_std_7d'] = agg_3d_df['pld_daily_mean'].rolling(3, min_periods=1).std().fillna(0)

# Target Delta (3-Day Delta)
agg_3d_df['target_delta'] = agg_3d_df['target_pld_next_day'] - agg_3d_df['pld_daily_mean']

# Drop any row with NaN
valid_mask_3d = agg_3d_df.notnull().all(axis=1)
agg_3d_df = agg_3d_df[valid_mask_3d].reset_index(drop=True)
bge_3d = bge_3d[valid_mask_3d].reset_index(drop=True)

# Split 3-Day Window (80% train, 20% test)
split_3d = int(len(agg_3d_df) * 0.8)
y_train_3d_d = agg_3d_df.iloc[:split_3d]['target_delta']
y_test_3d_d = agg_3d_df.iloc[split_3d:]['target_delta']
y_test_3d_lvl = agg_3d_df.iloc[split_3d:]['target_pld_next_day']
pld_base_3d = agg_3d_df.iloc[split_3d:]['pld_daily_mean']

# Supervised PLS & PCA on 3-day aggregated embeddings
pls_3d = PLSRegression(n_components=10)
pls_3d.fit(bge_3d.iloc[:split_3d], y_train_3d_d)
pls_3d_df = pd.DataFrame(pls_3d.transform(bge_3d), columns=[f'emb_pls_{i+1:02d}' for i in range(10)])

pca_3d = PCA(n_components=16, random_state=42)
pca_3d.fit(bge_3d.iloc[:split_3d])
pca_3d_df = pd.DataFrame(pca_3d.transform(bge_3d), columns=[f'emb_pca_{i+1:02d}' for i in range(16)])

feat_mkt_3d = [c for c in agg_3d_df.columns if c not in ['target_pld_next_day', 'target_delta']]
X_3d_mkt = agg_3d_df[feat_mkt_3d]
X_3d_full = pd.concat([X_3d_mkt, pls_3d_df, pca_3d_df], axis=1)
X_3d_news = pd.concat([agg_3d_df[['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']], pls_3d_df, pca_3d_df], axis=1)

print(f"3-Day Dataset: Total={len(agg_3d_df)} blocks | Train={split_3d} | Test={len(agg_3d_df)-split_3d}")

# Evaluate 3-Day Window Configurations
res_3d_mkt = train_and_eval(X_3d_mkt.iloc[:split_3d], X_3d_mkt.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)
res_3d_full = train_and_eval(X_3d_full.iloc[:split_3d], X_3d_full.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)
res_3d_news = train_and_eval(X_3d_news.iloc[:split_3d], X_3d_news.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)

# =============================================================
# PART 2: 12-HOUR TIME WINDOW PIPELINE (Peak vs Base 12-hour Load Blocks)
# =============================================================
print("\n" + "-"*60)
print("PROCESSING 12-HOUR TIME WINDOW (Sub-Daily Peak & Base Load Blocks)")
print("-"*60)

# Create 12-hour blocks (2 blocks per day: Peak / Heavy and Off-Peak / Light)
df_12h_peak = df_clean.copy()
df_12h_peak['period'] = 'Peak_12h'
df_12h_peak['block_pld'] = df_12h_peak['pld_daily_max']
df_12h_peak['block_cmo'] = df_12h_peak['cmo_heavy_load']
df_12h_peak['target_block_pld'] = df_12h_peak['target_pld_next_day'] * (df_12h_peak['pld_daily_max'] / (df_12h_peak['pld_daily_mean'] + 1e-3))

df_12h_base = df_clean.copy()
df_12h_base['period'] = 'Base_12h'
df_12h_base['block_pld'] = df_12h_base['pld_daily_min']
df_12h_base['block_cmo'] = df_12h_base['cmo_light_load']
df_12h_base['target_block_pld'] = df_12h_base['target_pld_next_day'] * (df_12h_base['pld_daily_min'] / (df_12h_base['pld_daily_mean'] + 1e-3))

# Interleave into 12-hour sequential steps
df_12h = pd.concat([df_12h_peak, df_12h_base]).sort_values(['Date_dt', 'period'], ascending=[True, False]).reset_index(drop=True)
bge_12h = pd.concat([bge_clean, bge_clean]).sort_index(kind='merge').reset_index(drop=True)

# 12-Hour Spreads & Dynamics
df_12h['intraday_spread'] = df_12h['pld_daily_max'] - df_12h['pld_daily_min']
df_12h['block_cmo_gap'] = df_12h['block_cmo'] - df_12h['block_pld']
df_12h['block_pld_lag1'] = df_12h['block_pld'].shift(1).fillna(df_12h['block_pld'])
df_12h['block_pld_lag2'] = df_12h['block_pld'].shift(2).fillna(df_12h['block_pld'])
df_12h['block_diff_lag1'] = df_12h['block_pld'] - df_12h['block_pld_lag1']
df_12h['block_rolling_std_6'] = df_12h['block_pld'].rolling(6, min_periods=1).std().fillna(0)

# Target Delta (12-Hour Delta)
df_12h['target_delta_12h'] = df_12h['target_block_pld'] - df_12h['block_pld']

# Drop any row with NaN
valid_mask_12h = df_12h.notnull().all(axis=1)
df_12h = df_12h[valid_mask_12h].reset_index(drop=True)
bge_12h = bge_12h[valid_mask_12h].reset_index(drop=True)

split_12h = int(len(df_12h) * 0.8)
y_train_12h_d = df_12h.iloc[:split_12h]['target_delta_12h']
y_test_12h_d = df_12h.iloc[split_12h:]['target_delta_12h']
y_test_12h_lvl = df_12h.iloc[split_12h:]['target_block_pld']
pld_base_12h = df_12h.iloc[split_12h:]['block_pld']

# PLS & PCA on 12-hour embeddings
pls_12h = PLSRegression(n_components=10)
pls_12h.fit(bge_12h.iloc[:split_12h], y_train_12h_d)
pls_12h_df = pd.DataFrame(pls_12h.transform(bge_12h), columns=[f'emb_pls_{i+1:02d}' for i in range(10)])

pca_12h = PCA(n_components=16, random_state=42)
pca_12h.fit(bge_12h.iloc[:split_12h])
pca_12h_df = pd.DataFrame(pca_12h.transform(bge_12h), columns=[f'emb_pca_{i+1:02d}' for i in range(16)])

feat_mkt_12h = [
    'block_pld', 'block_cmo', 'intraday_spread', 'block_cmo_gap',
    'block_pld_lag1', 'block_pld_lag2', 'block_diff_lag1', 'block_rolling_std_6',
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'cmo_weekly_mean'
]
X_12h_mkt = df_12h[feat_mkt_12h]
X_12h_full = pd.concat([df_12h[feat_mkt_12h + ['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']], pls_12h_df, pca_12h_df], axis=1)
X_12h_news = pd.concat([df_12h[['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']], pls_12h_df, pca_12h_df], axis=1)

print(f"12-Hour Dataset: Total={len(df_12h)} steps | Train={split_12h} | Test={len(df_12h)-split_12h}")

# Evaluate 12-Hour Window Configurations
res_12h_mkt = train_and_eval(X_12h_mkt.iloc[:split_12h], X_12h_mkt.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)
res_12h_full = train_and_eval(X_12h_full.iloc[:split_12h], X_12h_full.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)
res_12h_news = train_and_eval(X_12h_news.iloc[:split_12h], X_12h_news.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)

# =============================================================
# SUMMARY & COMPARISON ACROSS ALL TIME WINDOWS
# =============================================================
results_summary = {
    "12_Hour_Window": {
        "Market_Only": {k: v for k, v in res_12h_mkt.items() if k != "RF_Model"},
        "Full_Pipeline_With_Embeddings": {k: v for k, v in res_12h_full.items() if k != "RF_Model"},
        "News_Alone": {k: v for k, v in res_12h_news.items() if k != "RF_Model"}
    },
    "3_Day_Window": {
        "Market_Only": {k: v for k, v in res_3d_mkt.items() if k != "RF_Model"},
        "Full_Pipeline_With_Embeddings": {k: v for k, v in res_3d_full.items() if k != "RF_Model"},
        "News_Alone": {k: v for k, v in res_3d_news.items() if k != "RF_Model"}
    }
}

with open(get_result_path("window_benchmarks_results.json"), "w") as f:
    json.dump(results_summary, f, indent=4)

print("\n" + "="*100)
print("FINAL BENCHMARK COMPARISON ACROSS TIME WINDOWS (3-DAY vs 12-HOUR)")
print("="*100)
print(f"{'Time Window & Model Configuration':<45} | {'R2 Score':<10} | {'Expl. Var':<10} | {'RMSE (R$)':<10} | {'MAE (R$)':<10} | {'Dir. Acc %':<10}")
print("-" * 110)

print(f"{'[12-HOUR WINDOW] Market Only':<45} | {res_12h_mkt['R2']:.4f}{'':<4} | {res_12h_mkt['Explained_Variance']:.4f}{'':<4} | {res_12h_mkt['RMSE']:<10.4f} | {res_12h_mkt['MAE']:<10.4f} | {res_12h_mkt['Directional_Accuracy_Pct']:.2f}%")
print(f"{'[12-HOUR WINDOW] Full Pipeline (+ Embeddings)':<45} | {res_12h_full['R2']:.4f}{'':<4} | {res_12h_full['Explained_Variance']:.4f}{'':<4} | {res_12h_full['RMSE']:<10.4f} | {res_12h_full['MAE']:<10.4f} | {res_12h_full['Directional_Accuracy_Pct']:.2f}%")
print(f"{'[12-HOUR WINDOW] News Alone':<45} | {res_12h_news['R2']:.4f}{'':<4} | {res_12h_news['Explained_Variance']:.4f}{'':<4} | {res_12h_news['RMSE']:<10.4f} | {res_12h_news['MAE']:<10.4f} | {res_12h_news['Directional_Accuracy_Pct']:.2f}%")

print("-" * 110)
print(f"{'[3-DAY WINDOW]  Market Only':<45} | {res_3d_mkt['R2']:.4f}{'':<4} | {res_3d_mkt['Explained_Variance']:.4f}{'':<4} | {res_3d_mkt['RMSE']:<10.4f} | {res_3d_mkt['MAE']:<10.4f} | {res_3d_mkt['Directional_Accuracy_Pct']:.2f}%")
print(f"{'[3-DAY WINDOW]  Full Pipeline (+ Embeddings)':<45} | {res_3d_full['R2']:.4f}{'':<4} | {res_3d_full['Explained_Variance']:.4f}{'':<4} | {res_3d_full['RMSE']:<10.4f} | {res_3d_full['MAE']:<10.4f} | {res_3d_full['Directional_Accuracy_Pct']:.2f}%")
print(f"{'[3-DAY WINDOW]  News Alone':<45} | {res_3d_news['R2']:.4f}{'':<4} | {res_3d_news['Explained_Variance']:.4f}{'':<4} | {res_3d_news['RMSE']:<10.4f} | {res_3d_news['MAE']:<10.4f} | {res_3d_news['Directional_Accuracy_Pct']:.2f}%")
print("-" * 110)

print("\nBenchmark completed successfully! Saved to window_benchmarks_results.json")
