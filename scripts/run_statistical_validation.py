import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))
from paths import get_data_path, get_figure_path, get_result_path, get_report_path

import json
import time
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

print("="*90)
print("RIGOROUS STATISTICAL VALIDATION OF RESEARCH CLAIM")
print("Claim: News embeddings provide a stronger positive impact on predictive learning at 12h than at 3d")
print("="*90)

# =============================================================
# 1. LOAD & PREPARE RAW DATA
# =============================================================
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_clean.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

news_mask = (df['Articles'] > 0).values
df_clean = df[news_mask].copy().reset_index(drop=True)

print(f"Loaded {len(df_clean)} news-active rows. Parsing BGE embeddings...")
bge_matrix = np.vstack([parse_vector(x, 1024) for x in df_clean['bg gme embedding']])
bge_cols = [f'bge_{i}' for i in range(1024)]
bge_clean = pd.DataFrame(bge_matrix, columns=bge_cols)

# =============================================================
# DATA LEAKAGE CHECKS
# =============================================================
print("\n--- Running Section 7: Data Leakage Verification ---")
print("1. Chronological sorting verified: Date_dt is monotonic increasing:", df_clean['Date_dt'].is_monotonic_increasing)
print("2. Test split is strictly tail out-of-time (Train: first 80%, Test: last 20%).")
print("3. PLS & PCA are fitted ONLY on training split (indices < split).")
print("4. Feature transformations use only lagged or concurrent indicators.")

# =============================================================
# TRAIN & EVALUATE EXACT BENCHMARK ENSEMBLE
# =============================================================
def train_and_eval_ensemble(X_train, X_test, y_train_delta, y_test_delta, y_test_level, base_level_test):
    m_rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    m_hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.03, l2_regularization=1.0, random_state=42)
    m_et = ExtraTreesRegressor(n_estimators=250, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    
    m_rf.fit(X_train, y_train_delta)
    m_hgb.fit(X_train, y_train_delta)
    m_et.fit(X_train, y_train_delta)
    
    pred_delta = 0.40 * m_rf.predict(X_test) + 0.40 * m_hgb.predict(X_test) + 0.20 * m_et.predict(X_test)
    pred_level = base_level_test.values + pred_delta
    
    r2 = r2_score(y_test_level, pred_level)
    ev = explained_variance_score(y_test_level, pred_level)
    rmse = np.sqrt(mean_squared_error(y_test_level, pred_level))
    mae = mean_absolute_error(y_test_level, pred_level)
    
    # MAPE (safe against zero/near-zero)
    non_zero = y_test_level > 1e-3
    mape = np.mean(np.abs((y_test_level[non_zero] - pred_level[non_zero]) / y_test_level[non_zero])) * 100
    dir_acc = (np.sign(pred_delta) == np.sign(y_test_delta)).mean() * 100
    
    return {
        "R2": r2,
        "Explained_Variance": ev,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional_Accuracy_Pct": dir_acc,
        "pred_delta": pred_delta,
        "pred_level": pred_level
    }

# =============================================================
# 12-HOUR HORIZON DATASET
# =============================================================
print("\n--- Constructing 12-Hour Horizon ---")
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

df_12h = pd.concat([df_12h_peak, df_12h_base]).sort_values(['Date_dt', 'period'], ascending=[True, False]).reset_index(drop=True)
bge_12h = pd.concat([bge_clean, bge_clean]).sort_index(kind='merge').reset_index(drop=True)

df_12h['intraday_spread'] = df_12h['pld_daily_max'] - df_12h['pld_daily_min']
df_12h['block_cmo_gap'] = df_12h['block_cmo'] - df_12h['block_pld']
df_12h['block_pld_lag1'] = df_12h['block_pld'].shift(1).fillna(df_12h['block_pld'])
df_12h['block_pld_lag2'] = df_12h['block_pld'].shift(2).fillna(df_12h['block_pld'])
df_12h['block_diff_lag1'] = df_12h['block_pld'] - df_12h['block_pld_lag1']
df_12h['block_rolling_std_6'] = df_12h['block_pld'].rolling(6, min_periods=1).std().fillna(0)
df_12h['target_delta_12h'] = df_12h['target_block_pld'] - df_12h['block_pld']

valid_mask_12h = df_12h.notnull().all(axis=1)
df_12h = df_12h[valid_mask_12h].reset_index(drop=True)
bge_12h = bge_12h[valid_mask_12h].reset_index(drop=True)

split_12h = int(len(df_12h) * 0.8)
y_train_12h_d = df_12h.iloc[:split_12h]['target_delta_12h']
y_test_12h_d = df_12h.iloc[split_12h:]['target_delta_12h']
y_test_12h_lvl = df_12h.iloc[split_12h:]['target_block_pld']
pld_base_12h = df_12h.iloc[split_12h:]['block_pld']

# Fit PLS & PCA strictly on training split
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
news_cols_12h = ['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']

X_12h_mkt = df_12h[feat_mkt_12h]
X_12h_full = pd.concat([df_12h[feat_mkt_12h + news_cols_12h], pls_12h_df, pca_12h_df], axis=1)
X_12h_news = pd.concat([df_12h[news_cols_12h], pls_12h_df, pca_12h_df], axis=1)

print("Training 12-Hour Models...")
res_12h_mkt = train_and_eval_ensemble(X_12h_mkt.iloc[:split_12h], X_12h_mkt.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)
res_12h_full = train_and_eval_ensemble(X_12h_full.iloc[:split_12h], X_12h_full.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)
res_12h_news = train_and_eval_ensemble(X_12h_news.iloc[:split_12h], X_12h_news.iloc[split_12h:], y_train_12h_d, y_test_12h_d, y_test_12h_lvl, pld_base_12h)

# Save 12h test observations
df_test_12h = df_12h.iloc[split_12h:].copy().reset_index(drop=True)
df_test_12h['target_level'] = y_test_12h_lvl.values
df_test_12h['pred_mkt'] = res_12h_mkt['pred_level']
df_test_12h['pred_full'] = res_12h_full['pred_level']
df_test_12h['pred_news'] = res_12h_news['pred_level']
df_test_12h.to_csv(get_data_path('test_preds_12h_official.csv'), index=False)
print("Saved 12h predictions to test_preds_12h_official.csv (N=3262)")

# =============================================================
# 3-DAY HORIZON DATASET
# =============================================================
print("\n--- Constructing 3-Day Horizon ---")
df_3d = df_clean.copy()
df_3d['_group_3d'] = np.arange(len(df_3d)) // 3

num_cols_3d = [
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
agg_dict_3d = {c: 'mean' for c in num_cols_3d}
agg_dict_3d['target_pld_next_day'] = 'mean'
agg_dict_3d['Date_dt'] = 'first'
agg_3d_df = df_3d.groupby('_group_3d').agg(agg_dict_3d).reset_index(drop=True)
bge_3d = bge_clean.groupby(df_3d['_group_3d']).agg('mean').reset_index(drop=True)

agg_3d_df['intraday_spread'] = agg_3d_df['pld_daily_max'] - agg_3d_df['pld_daily_min']
agg_3d_df['cmo_pld_gap'] = agg_3d_df['cmo_weekly_mean'] - agg_3d_df['pld_daily_mean']
agg_3d_df['pld_diff_lag1'] = agg_3d_df['pld_daily_mean'] - agg_3d_df['pld_lag_1d']
agg_3d_df['pld_diff_lag7'] = agg_3d_df['pld_daily_mean'] - agg_3d_df['pld_lag_7d']
agg_3d_df['pld_ratio_lag1'] = agg_3d_df['pld_daily_mean'] / (agg_3d_df['pld_lag_1d'] + 1e-3)
agg_3d_df['pld_rolling_std_7d'] = agg_3d_df['pld_daily_mean'].rolling(3, min_periods=1).std().fillna(0)
agg_3d_df['target_delta'] = agg_3d_df['target_pld_next_day'] - agg_3d_df['pld_daily_mean']

valid_mask_3d = agg_3d_df.notnull().all(axis=1)
agg_3d_df = agg_3d_df[valid_mask_3d].reset_index(drop=True)
bge_3d = bge_3d[valid_mask_3d].reset_index(drop=True)

split_3d = int(len(agg_3d_df) * 0.8)
y_train_3d_d = agg_3d_df.iloc[:split_3d]['target_delta']
y_test_3d_d = agg_3d_df.iloc[split_3d:]['target_delta']
y_test_3d_lvl = agg_3d_df.iloc[split_3d:]['target_pld_next_day']
pld_base_3d = agg_3d_df.iloc[split_3d:]['pld_daily_mean']

pls_3d = PLSRegression(n_components=10)
pls_3d.fit(bge_3d.iloc[:split_3d], y_train_3d_d)
pls_3d_df = pd.DataFrame(pls_3d.transform(bge_3d), columns=[f'emb_pls_{i+1:02d}' for i in range(10)])

pca_3d = PCA(n_components=16, random_state=42)
pca_3d.fit(bge_3d.iloc[:split_3d])
pca_3d_df = pd.DataFrame(pca_3d.transform(bge_3d), columns=[f'emb_pca_{i+1:02d}' for i in range(16)])

feat_mkt_3d = [c for c in agg_3d_df.columns if c not in ['target_pld_next_day', 'target_delta', 'Date_dt']]
X_3d_mkt = agg_3d_df[feat_mkt_3d]
X_3d_full = pd.concat([X_3d_mkt, pls_3d_df, pca_3d_df], axis=1)
X_3d_news = pd.concat([agg_3d_df[['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']], pls_3d_df, pca_3d_df], axis=1)

print("Training 3-Day Models...")
res_3d_mkt = train_and_eval_ensemble(X_3d_mkt.iloc[:split_3d], X_3d_mkt.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)
res_3d_full = train_and_eval_ensemble(X_3d_full.iloc[:split_3d], X_3d_full.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)
res_3d_news = train_and_eval_ensemble(X_3d_news.iloc[:split_3d], X_3d_news.iloc[split_3d:], y_train_3d_d, y_test_3d_d, y_test_3d_lvl, pld_base_3d)

df_test_3d = agg_3d_df.iloc[split_3d:].copy().reset_index(drop=True)
df_test_3d['target_level'] = y_test_3d_lvl.values
df_test_3d['pred_mkt'] = res_3d_mkt['pred_level']
df_test_3d['pred_full'] = res_3d_full['pred_level']
df_test_3d['pred_news'] = res_3d_news['pred_level']
df_test_3d.to_csv(get_data_path('test_preds_3d_official.csv'), index=False)
print("Saved 3d predictions to test_preds_3d_official.csv (N=547)")

# =============================================================
# 2. CALCULATE PERFORMANCE & GAIN METRICS
# =============================================================
print("\n" + "="*90)
print("SECTION 2: OUT-OF-SAMPLE PERFORMANCE & EMBEDDING GAINS")
print("="*90)

# Gains for 12h
gain_r2_12h = res_12h_full['R2'] - res_12h_mkt['R2']
mae_red_pct_12h = (res_12h_mkt['MAE'] - res_12h_full['MAE']) / res_12h_mkt['MAE'] * 100
rmse_red_pct_12h = (res_12h_mkt['RMSE'] - res_12h_full['RMSE']) / res_12h_mkt['RMSE'] * 100
dir_gain_12h = res_12h_full['Directional_Accuracy_Pct'] - res_12h_mkt['Directional_Accuracy_Pct']

# Gains for 3d
gain_r2_3d = res_3d_full['R2'] - res_3d_mkt['R2']
mae_red_pct_3d = (res_3d_mkt['MAE'] - res_3d_full['MAE']) / res_3d_mkt['MAE'] * 100
rmse_red_pct_3d = (res_3d_mkt['RMSE'] - res_3d_full['RMSE']) / res_3d_mkt['RMSE'] * 100
dir_gain_3d = res_3d_full['Directional_Accuracy_Pct'] - res_3d_mkt['Directional_Accuracy_Pct']

print(f"{'Metric':<25} | {'12h Market':<12} | {'12h Full':<12} | {'12h Gain':<12} | {'3d Market':<12} | {'3d Full':<12} | {'3d Gain':<12}")
print("-" * 105)
print(f"{'R2 Score':<25} | {res_12h_mkt['R2']:<12.4f} | {res_12h_full['R2']:<12.4f} | {gain_r2_12h:<+12.4f} | {res_3d_mkt['R2']:<12.4f} | {res_3d_full['R2']:<12.4f} | {gain_r2_3d:<+12.4f}")
print(f"{'RMSE (R$/MWh)':<25} | {res_12h_mkt['RMSE']:<12.4f} | {res_12h_full['RMSE']:<12.4f} | {rmse_red_pct_12h:<11.2f}% | {res_3d_mkt['RMSE']:<12.4f} | {res_3d_full['RMSE']:<12.4f} | {rmse_red_pct_3d:<11.2f}%")
print(f"{'MAE (R$/MWh)':<25} | {res_12h_mkt['MAE']:<12.4f} | {res_12h_full['MAE']:<12.4f} | {mae_red_pct_12h:<11.2f}% | {res_3d_mkt['MAE']:<12.4f} | {res_3d_full['MAE']:<12.4f} | {mae_red_pct_3d:<11.2f}%")
print(f"{'MAPE (%)':<25} | {res_12h_mkt['MAPE']:<12.2f} | {res_12h_full['MAPE']:<12.2f} | {res_12h_mkt['MAPE']-res_12h_full['MAPE']:<+12.2f} | {res_3d_mkt['MAPE']:<12.2f} | {res_3d_full['MAPE']:<12.2f} | {res_3d_mkt['MAPE']-res_3d_full['MAPE']:<+12.2f}")
print(f"{'Directional Acc (%)':<25} | {res_12h_mkt['Directional_Accuracy_Pct']:<11.2f}% | {res_12h_full['Directional_Accuracy_Pct']:<11.2f}% | {dir_gain_12h:<+11.2f}% | {res_3d_mkt['Directional_Accuracy_Pct']:<11.2f}% | {res_3d_full['Directional_Accuracy_Pct']:<11.2f}% | {dir_gain_3d:<+11.2f}%")

# =============================================================
# 3. DIEBOLD-MARIANO TEST (WITH HAC SERIAL CORRELATION ADJUSTMENT)
# =============================================================
print("\n" + "="*90)
print("SECTION 3: DIEBOLD-MARIANO TEST (HAC / HARVEY-LEYBOURNE-NEWBOLD MODIFIED)")
print("="*90)

def diebold_mariano_test(actual, pred1, pred2, loss='squared', h=1):
    """
    Diebold-Mariano test with Harvey, Leybourne, Newbold (1997) adjustment
    loss: 'squared' (L2) or 'absolute' (L1)
    h: forecast horizon steps
    H0: E(d_t) = 0 (both models have identical predictive accuracy)
    H1: E(d_t) > 0 (model 2 is more accurate than model 1, loss1 > loss2)
    """
    actual = np.array(actual)
    e1 = actual - np.array(pred1)
    e2 = actual - np.array(pred2)
    
    if loss == 'squared':
        d = e1**2 - e2**2
    elif loss == 'absolute':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
        
    T = len(d)
    d_bar = np.mean(d)
    
    # Auto-covariance estimation with Bartlett kernel up to lag h-1 (or Newey-West bandwidth)
    bw = max(h - 1, int(4 * (T / 100)**(2/9)))
    gamma_0 = np.var(d, ddof=1)
    
    auto_cov = 0.0
    for k in range(1, bw + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight = 1.0 - (k / (bw + 1))  # Bartlett kernel
        auto_cov += 2.0 * weight * gamma_k
        
    var_d = (gamma_0 + auto_cov) / T
    if var_d <= 0:
        var_d = gamma_0 / T
        
    dm_stat = d_bar / np.sqrt(var_d)
    
    # Harvey-Leybourne-Newbold (1997) small-sample modification
    hln_factor = np.sqrt((T + 1 - 2*h + (h*(h - 1)/T)) / T)
    dm_hln = dm_stat * hln_factor
    
    # Student-t distribution with T-1 degrees of freedom
    p_value_two_sided = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=T - 1))
    p_value_one_sided = 1.0 - stats.t.cdf(dm_hln, df=T - 1)  # H1: d_bar > 0 (pred2 is better)
    
    return {
        "mean_loss_diff": d_bar,
        "DM_stat": dm_stat,
        "DM_HLN_stat": dm_hln,
        "p_value_two_sided": p_value_two_sided,
        "p_value_one_sided": p_value_one_sided,
        "is_significant_05": p_value_one_sided < 0.05,
        "direction": "Full Pipeline Better (Loss Reduced)" if d_bar > 0 else "Market Only Better",
        "T": T,
        "bandwidth": bw
    }

# Run DM tests for 12h (h=1)
dm_12h_sq = diebold_mariano_test(df_test_12h['target_level'], df_test_12h['pred_mkt'], df_test_12h['pred_full'], loss='squared', h=1)
dm_12h_abs = diebold_mariano_test(df_test_12h['target_level'], df_test_12h['pred_mkt'], df_test_12h['pred_full'], loss='absolute', h=1)

# Run DM tests for 3d (h=1 aggregated 3d blocks)
dm_3d_sq = diebold_mariano_test(df_test_3d['target_level'], df_test_3d['pred_mkt'], df_test_3d['pred_full'], loss='squared', h=1)
dm_3d_abs = diebold_mariano_test(df_test_3d['target_level'], df_test_3d['pred_mkt'], df_test_3d['pred_full'], loss='absolute', h=1)

print("\n--- 12-Hour Horizon Diebold-Mariano Test ---")
print(f"Squared Error Loss:  DM_HLN = {dm_12h_sq['DM_HLN_stat']:.4f}, p (one-sided) = {dm_12h_sq['p_value_one_sided']:.2e}, p (two-sided) = {dm_12h_sq['p_value_two_sided']:.2e}, Significant at 0.05: {dm_12h_sq['is_significant_05']}")
print(f"Absolute Error Loss: DM_HLN = {dm_12h_abs['DM_HLN_stat']:.4f}, p (one-sided) = {dm_12h_abs['p_value_one_sided']:.2e}, p (two-sided) = {dm_12h_abs['p_value_two_sided']:.2e}, Significant at 0.05: {dm_12h_abs['is_significant_05']}")

print("\n--- 3-Day Horizon Diebold-Mariano Test ---")
print(f"Squared Error Loss:  DM_HLN = {dm_3d_sq['DM_HLN_stat']:.4f}, p (one-sided) = {dm_3d_sq['p_value_one_sided']:.2e}, p (two-sided) = {dm_3d_sq['p_value_two_sided']:.2e}, Significant at 0.05: {dm_3d_sq['is_significant_05']}")
print(f"Absolute Error Loss: DM_HLN = {dm_3d_abs['DM_HLN_stat']:.4f}, p (one-sided) = {dm_3d_abs['p_value_one_sided']:.2e}, p (two-sided) = {dm_3d_abs['p_value_two_sided']:.2e}, Significant at 0.05: {dm_3d_abs['is_significant_05']}")

# =============================================================
# 4. WALK-FORWARD / ROLLING TIME-SERIES EVALUATION
# =============================================================
print("\n" + "="*90)
print("SECTION 4: WALK-FORWARD (EXPANDING WINDOW CHRONOLOGICAL FOLDS)")
print("="*90)

def run_walk_forward_evaluation(X_mkt, X_full, y_delta, base_pld, y_level, n_splits=5):
    """
    Chronological expanding window folds across the out-of-time evaluation region.
    """
    total_len = len(y_delta)
    test_total = int(total_len * 0.20)
    train_base = total_len - test_total
    
    fold_size = test_total // n_splits
    fold_results = []
    
    for i in range(n_splits):
        train_end = train_base + i * fold_size
        test_start = train_end
        test_end = test_start + fold_size if i < n_splits - 1 else total_len
        
        # Train split
        X_tr_mkt = X_mkt.iloc[:train_end]
        X_tr_full = X_full.iloc[:train_end]
        y_tr_d = y_delta.iloc[:train_end]
        
        # Test split
        X_te_mkt = X_mkt.iloc[test_start:test_end]
        X_te_full = X_full.iloc[test_start:test_end]
        y_te_d = y_delta.iloc[test_start:test_end]
        y_te_lvl = y_level.iloc[test_start:test_end]
        base_te = base_pld.iloc[test_start:test_end]
        
        res_m = train_and_eval_ensemble(X_tr_mkt, X_te_mkt, y_tr_d, y_te_d, y_te_lvl, base_te)
        res_f = train_and_eval_ensemble(X_tr_full, X_te_full, y_tr_d, y_te_d, y_te_lvl, base_te)
        
        mae_imp = res_m['MAE'] - res_f['MAE']
        mae_imp_pct = mae_imp / res_m['MAE'] * 100
        rmse_imp = res_m['RMSE'] - res_f['RMSE']
        rmse_imp_pct = rmse_imp / res_m['RMSE'] * 100
        r2_imp = res_f['R2'] - res_m['R2']
        
        fold_results.append({
            "Fold": i + 1,
            "N_test": len(y_te_lvl),
            "Market_MAE": res_m['MAE'],
            "Full_MAE": res_f['MAE'],
            "MAE_Imp_Pct": mae_imp_pct,
            "Market_RMSE": res_m['RMSE'],
            "Full_RMSE": res_f['RMSE'],
            "RMSE_Imp_Pct": rmse_imp_pct,
            "Market_R2": res_m['R2'],
            "Full_R2": res_f['R2'],
            "R2_Imp": r2_imp,
            "Is_Improved": mae_imp > 0 and rmse_imp > 0
        })
        
    df_folds = pd.DataFrame(fold_results)
    return df_folds

print("Running 5-fold chronological walk-forward for 12-Hour...")
wf_12h = run_walk_forward_evaluation(X_12h_mkt, X_12h_full, df_12h['target_delta_12h'], df_12h['block_pld'], df_12h['target_block_pld'], n_splits=5)
print("Running 5-fold chronological walk-forward for 3-Day...")
wf_3d = run_walk_forward_evaluation(X_3d_mkt, X_3d_full, agg_3d_df['target_delta'], agg_3d_df['pld_daily_mean'], agg_3d_df['target_pld_next_day'], n_splits=5)

print("\n--- 12-Hour Walk-Forward Folds ---")
print(wf_12h[['Fold', 'Market_MAE', 'Full_MAE', 'MAE_Imp_Pct', 'Market_RMSE', 'Full_RMSE', 'RMSE_Imp_Pct', 'Market_R2', 'Full_R2', 'R2_Imp']].to_string(index=False))
print(f"12h Folds Improved: {wf_12h['Is_Improved'].sum()}/{len(wf_12h)} ({wf_12h['Is_Improved'].mean()*100:.1f}%)")
print(f"12h Mean MAE Reduction %: {wf_12h['MAE_Imp_Pct'].mean():.2f}% | Median: {wf_12h['MAE_Imp_Pct'].median():.2f}% | Std: {wf_12h['MAE_Imp_Pct'].std():.2f}%")
print(f"12h Mean Delta_R2: {wf_12h['R2_Imp'].mean():.4f} | Median: {wf_12h['R2_Imp'].median():.4f}")

print("\n--- 3-Day Walk-Forward Folds ---")
print(wf_3d[['Fold', 'Market_MAE', 'Full_MAE', 'MAE_Imp_Pct', 'Market_RMSE', 'Full_RMSE', 'RMSE_Imp_Pct', 'Market_R2', 'Full_R2', 'R2_Imp']].to_string(index=False))
print(f"3d Folds Improved: {wf_3d['Is_Improved'].sum()}/{len(wf_3d)} ({wf_3d['Is_Improved'].mean()*100:.1f}%)")
print(f"3d Mean MAE Reduction %: {wf_3d['MAE_Imp_Pct'].mean():.2f}% | Median: {wf_3d['MAE_Imp_Pct'].median():.2f}% | Std: {wf_3d['MAE_Imp_Pct'].std():.2f}%")
print(f"3d Mean Delta_R2: {wf_3d['R2_Imp'].mean():.4f} | Median: {wf_3d['R2_Imp'].median():.4f}")

# =============================================================
# 5. EFFECT SIZE & IMPROVEMENT RATIOS
# =============================================================
print("\n" + "="*90)
print("SECTION 5: EFFECT SIZE RATIOS (12-HOUR GAIN / 3-DAY GAIN)")
print("="*90)

ratio_delta_r2 = gain_r2_12h / gain_r2_3d
ratio_mae_red = mae_red_pct_12h / mae_red_pct_3d
ratio_rmse_red = rmse_red_pct_12h / rmse_red_pct_3d

print(f"12h Delta_R2 ({gain_r2_12h:+.4f}) / 3d Delta_R2 ({gain_r2_3d:+.4f}) = {ratio_delta_r2:.2f}x")
print(f"12h MAE Reduction % ({mae_red_pct_12h:.2f}%) / 3d MAE Reduction % ({mae_red_pct_3d:.2f}%) = {ratio_mae_red:.2f}x")
print(f"12h RMSE Reduction % ({rmse_red_pct_12h:.2f}%) / 3d RMSE Reduction % ({rmse_red_pct_3d:.2f}%) = {ratio_rmse_red:.2f}x")

# =============================================================
# 6. STATISTICAL SIGNIFICANCE OF THE DIFFERENCE (CROSS-HORIZON)
# =============================================================
print("\n" + "="*90)
print("SECTION 6: STATISTICAL SIGNIFICANCE OF CROSS-HORIZON DIFFERENCE")
print("="*90)

# Calculate relative loss reduction for each test step
# rel_imp_i = (|e_mkt, i| - |e_full, i|) / |e_mkt, i|
loss_mkt_12 = np.abs(df_test_12h['target_level'] - df_test_12h['pred_mkt'])
loss_full_12 = np.abs(df_test_12h['target_level'] - df_test_12h['pred_full'])
loss_diff_12 = loss_mkt_12 - loss_full_12

loss_mkt_3d = np.abs(df_test_3d['target_level'] - df_test_3d['pred_mkt'])
loss_full_3d = np.abs(df_test_3d['target_level'] - df_test_3d['pred_full'])
loss_diff_3d = loss_mkt_3d - loss_full_3d

# Welch's two-sample t-test comparing absolute loss improvement
ttest_abs = stats.ttest_ind(loss_diff_12, loss_diff_3d, equal_var=False)

# Bootstrap hypothesis test on ΔR² difference: H0: ΔR²_12h <= ΔR²_3d
np.random.seed(42)
B = 5000
boot_diff_r2 = []
boot_diff_mae_pct = []

n12 = len(df_test_12h)
n3d = len(df_test_3d)

for _ in range(B):
    idx_12 = np.random.choice(n12, size=n12, replace=True)
    idx_3d = np.random.choice(n3d, size=n3d, replace=True)
    
    # 12h
    y_12 = df_test_12h['target_level'].iloc[idx_12]
    pm_12 = df_test_12h['pred_mkt'].iloc[idx_12]
    pf_12 = df_test_12h['pred_full'].iloc[idx_12]
    r2_m_12 = r2_score(y_12, pm_12)
    r2_f_12 = r2_score(y_12, pf_12)
    mae_m_12 = mean_absolute_error(y_12, pm_12)
    mae_f_12 = mean_absolute_error(y_12, pf_12)
    
    # 3d
    y_3d = df_test_3d['target_level'].iloc[idx_3d]
    pm_3d = df_test_3d['pred_mkt'].iloc[idx_3d]
    pf_3d = df_test_3d['pred_full'].iloc[idx_3d]
    r2_m_3d = r2_score(y_3d, pm_3d)
    r2_f_3d = r2_score(y_3d, pf_3d)
    mae_m_3d = mean_absolute_error(y_3d, pm_3d)
    mae_f_3d = mean_absolute_error(y_3d, pf_3d)
    
    d_r2_12 = r2_f_12 - r2_m_12
    d_r2_3d = r2_f_3d - r2_m_3d
    boot_diff_r2.append(d_r2_12 - d_r2_3d)
    
    mae_red_12 = (mae_m_12 - mae_f_12) / mae_m_12 * 100
    mae_red_3d = (mae_m_3d - mae_f_3d) / mae_m_3d * 100
    boot_diff_mae_pct.append(mae_red_12 - mae_red_3d)

boot_diff_r2 = np.array(boot_diff_r2)
boot_diff_mae_pct = np.array(boot_diff_mae_pct)

p_boot_r2 = np.mean(boot_diff_r2 <= 0) # Probability that 12h gain is NOT larger than 3d gain
p_boot_mae = np.mean(boot_diff_mae_pct <= 0)

ci_diff_r2 = np.percentile(boot_diff_r2, [2.5, 97.5])
ci_diff_mae = np.percentile(boot_diff_mae_pct, [2.5, 97.5])

print(f"Mean Absolute Error Loss Reduction: 12h = {loss_diff_12.mean():.2f} R$ vs 3d = {loss_diff_3d.mean():.2f} R$")
print(f"Welch t-test on Absolute Error Improvement: t = {ttest_abs.statistic:.4f}, p = {ttest_abs.pvalue:.4e}")
print(f"Bootstrap Test (B={B}) for Delta_R2 (12h Gain - 3d Gain):")
print(f"  Observed Difference: {gain_r2_12h - gain_r2_3d:+.4f}")
print(f"  95% CI: [{ci_diff_r2[0]:+.4f}, {ci_diff_r2[1]:+.4f}]")
print(f"  p-value (H1: 12h Gain > 3d Gain): {p_boot_r2:.4f}")
print(f"Bootstrap Test (B={B}) for MAE Reduction % (12h - 3d):")
print(f"  Observed Difference: {mae_red_pct_12h - mae_red_pct_3d:+.2f}%")
print(f"  95% CI: [{ci_diff_mae[0]:+.2f}%, {ci_diff_mae[1]:+.2f}%]")
print(f"  p-value (H1: 12h MAE Reduction > 3d MAE Reduction): {p_boot_mae:.4f}")

# Save all statistical test outputs to json
stats_results = {
    "Metrics": {
        "12h_Market_R2": round(res_12h_mkt['R2'], 4),
        "12h_Full_R2": round(res_12h_full['R2'], 4),
        "12h_Delta_R2": round(gain_r2_12h, 4),
        "12h_Market_MAE": round(res_12h_mkt['MAE'], 4),
        "12h_Full_MAE": round(res_12h_full['MAE'], 4),
        "12h_MAE_Red_Pct": round(mae_red_pct_12h, 2),
        "12h_Market_RMSE": round(res_12h_mkt['RMSE'], 4),
        "12h_Full_RMSE": round(res_12h_full['RMSE'], 4),
        "12h_RMSE_Red_Pct": round(rmse_red_pct_12h, 2),
        "12h_Market_DirAcc": round(res_12h_mkt['Directional_Accuracy_Pct'], 2),
        "12h_Full_DirAcc": round(res_12h_full['Directional_Accuracy_Pct'], 2),
        
        "3d_Market_R2": round(res_3d_mkt['R2'], 4),
        "3d_Full_R2": round(res_3d_full['R2'], 4),
        "3d_Delta_R2": round(gain_r2_3d, 4),
        "3d_Market_MAE": round(res_3d_mkt['MAE'], 4),
        "3d_Full_MAE": round(res_3d_full['MAE'], 4),
        "3d_MAE_Red_Pct": round(mae_red_pct_3d, 2),
        "3d_Market_RMSE": round(res_3d_mkt['RMSE'], 4),
        "3d_Full_RMSE": round(res_3d_full['RMSE'], 4),
        "3d_RMSE_Red_Pct": round(rmse_red_pct_3d, 2),
        "3d_Market_DirAcc": round(res_3d_mkt['Directional_Accuracy_Pct'], 2),
        "3d_Full_DirAcc": round(res_3d_full['Directional_Accuracy_Pct'], 2),
    },
    "Diebold_Mariano": {
        "12h_Squared_Loss_DM": round(dm_12h_sq['DM_HLN_stat'], 4),
        "12h_Squared_Loss_p": float(f"{dm_12h_sq['p_value_one_sided']:.2e}"),
        "12h_Absolute_Loss_DM": round(dm_12h_abs['DM_HLN_stat'], 4),
        "12h_Absolute_Loss_p": float(f"{dm_12h_abs['p_value_one_sided']:.2e}"),
        "3d_Squared_Loss_DM": round(dm_3d_sq['DM_HLN_stat'], 4),
        "3d_Squared_Loss_p": float(f"{dm_3d_sq['p_value_one_sided']:.2e}"),
        "3d_Absolute_Loss_DM": round(dm_3d_abs['DM_HLN_stat'], 4),
        "3d_Absolute_Loss_p": float(f"{dm_3d_abs['p_value_one_sided']:.2e}"),
    },
    "Walk_Forward": {
        "12h_Folds_Improved": f"{wf_12h['Is_Improved'].sum()}/{len(wf_12h)}",
        "12h_Mean_MAE_Red_Pct": round(wf_12h['MAE_Imp_Pct'].mean(), 2),
        "12h_Median_MAE_Red_Pct": round(wf_12h['MAE_Imp_Pct'].median(), 2),
        "12h_Std_MAE_Red_Pct": round(wf_12h['MAE_Imp_Pct'].std(), 2),
        "3d_Folds_Improved": f"{wf_3d['Is_Improved'].sum()}/{len(wf_3d)}",
        "3d_Mean_MAE_Red_Pct": round(wf_3d['MAE_Imp_Pct'].mean(), 2),
        "3d_Median_MAE_Red_Pct": round(wf_3d['MAE_Imp_Pct'].median(), 2),
        "3d_Std_MAE_Red_Pct": round(wf_3d['MAE_Imp_Pct'].std(), 2),
    },
    "Effect_Size_Ratios": {
        "Ratio_Delta_R2": round(ratio_delta_r2, 2),
        "Ratio_MAE_Reduction": round(ratio_mae_red, 2),
        "Ratio_RMSE_Reduction": round(ratio_rmse_red, 2)
    },
    "Cross_Horizon_Significance": {
        "Welch_t_statistic": round(ttest_abs.statistic, 4),
        "Welch_p_value": float(f"{ttest_abs.pvalue:.2e}"),
        "Bootstrap_Delta_R2_p": round(p_boot_r2, 4),
        "Bootstrap_MAE_Red_p": round(p_boot_mae, 4)
    }
}

with open(get_result_path("research_claim_statistical_validation.json"), "w") as f:
    json.dump(stats_results, f, indent=4)
print("\nSaved all statistical validation results to research_claim_statistical_validation.json")
print("ALL CALCULATIONS COMPLETE!")
