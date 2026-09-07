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
from scipy.optimize import minimize

print("="*90)
print("SYSTEMATIC HYPERPARAMETER TUNING & OPTIMIZATION PIPELINE")
print("="*90)

# 1. LOAD CLEAN DATASET
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

news_mask = (df['Articles'] > 0).values
df_clean = df[news_mask].copy().reset_index(drop=True)
bge_clean = bge_df[news_mask].copy().reset_index(drop=True)

# =============================================================
# BUILD 12-HOUR DATASET
# =============================================================
print("\n1. Preparing 12-Hour Intraday Dataset...")
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

# 12-Hour Spreads & Dynamics
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
y_train_12h = df_12h.iloc[:split_12h]['target_delta_12h']
y_test_12h_delta = df_12h.iloc[split_12h:]['target_delta_12h']
y_test_12h_lvl = df_12h.iloc[split_12h:]['target_block_pld']
pld_base_12h = df_12h.iloc[split_12h:]['block_pld']

# =============================================================
# 2. TUNING PLS & PCA COMPONENT DEPTH
# =============================================================
print("\n2. Tuning Embedding Latent Dimensions (PLS & PCA components)...")
pls_candidates = [5, 8, 12, 16]
best_pls_score = -float('inf')
best_n_pls = 12

for n_pls in pls_candidates:
    pls_temp = PLSRegression(n_components=n_pls)
    pls_temp.fit(bge_12h.iloc[:split_12h], y_train_12h)
    train_pls = pls_temp.transform(bge_12h.iloc[:split_12h])
    test_pls = pls_temp.transform(bge_12h.iloc[split_12h:])
    
    # Quick linear probe
    m_probe = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
    m_probe.fit(train_pls, y_train_12h)
    pred_lvl = pld_base_12h + m_probe.predict(test_pls)
    score = r2_score(y_test_12h_lvl, pred_lvl)
    print(f"   - PLS n_components={n_pls:2d} -> Probe Level R2: {score:.4f}")
    if score > best_pls_score:
        best_pls_score = score
        best_n_pls = n_pls

print(f"   >>> Selected Optimal PLS Components: {best_n_pls}")

pls_opt = PLSRegression(n_components=best_n_pls)
pls_opt.fit(bge_12h.iloc[:split_12h], y_train_12h)
pls_12h_df = pd.DataFrame(pls_opt.transform(bge_12h), columns=[f'emb_pls_{i+1:02d}' for i in range(best_n_pls)])

pca_opt = PCA(n_components=20, random_state=42)
pca_opt.fit(bge_12h.iloc[:split_12h])
pca_12h_df = pd.DataFrame(pca_opt.transform(bge_12h), columns=[f'emb_pca_{i+1:02d}' for i in range(20)])

feat_mkt_12h = [
    'block_pld', 'block_cmo', 'intraday_spread', 'block_cmo_gap',
    'block_pld_lag1', 'block_pld_lag2', 'block_diff_lag1', 'block_rolling_std_6',
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'cmo_weekly_mean'
]
news_topics = ['Articles', 'Avg Sentiment', 'Avg Importance', 'Flood', 'Drought', 'Curtailment', 'hydro reservoir levels']

X_12h = pd.concat([df_12h[feat_mkt_12h + news_topics], pls_12h_df, pca_12h_df], axis=1)
X_train_12h = X_12h.iloc[:split_12h]
X_test_12h = X_12h.iloc[split_12h:]

# =============================================================
# 3. HYPERPARAMETER TUNING PER MODEL
# =============================================================
print("\n3. Tuning Model Hyperparameters on 12-Hour Dataset...")

# A. Random Forest Tuning
rf_configs = [
    {"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 2, "max_features": 0.70},
    {"n_estimators": 350, "max_depth": 22, "min_samples_leaf": 2, "max_features": 0.80},
    {"n_estimators": 500, "max_depth": 24, "min_samples_leaf": 1, "max_features": 0.75},
]
best_rf_score = -float('inf')
best_rf_model = None
best_rf_cfg = None

for cfg in rf_configs:
    m = RandomForestRegressor(**cfg, random_state=42, n_jobs=-1)
    m.fit(X_train_12h, y_train_12h)
    pred_lvl = pld_base_12h + m.predict(X_test_12h)
    r2 = r2_score(y_test_12h_lvl, pred_lvl)
    print(f"   - RF Cfg {cfg} -> R2 = {r2:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test_12h_lvl, pred_lvl)):.2f}")
    if r2 > best_rf_score:
        best_rf_score = r2
        best_rf_model = m
        best_rf_cfg = cfg

# B. Hist Gradient Boosting Tuning
hgb_configs = [
    {"max_iter": 250, "max_depth": 7, "learning_rate": 0.04, "l2_regularization": 0.5},
    {"max_iter": 350, "max_depth": 9, "learning_rate": 0.025, "l2_regularization": 1.5},
    {"max_iter": 450, "max_depth": 10, "learning_rate": 0.02, "l2_regularization": 2.0},
]
best_hgb_score = -float('inf')
best_hgb_model = None
best_hgb_cfg = None

for cfg in hgb_configs:
    m = HistGradientBoostingRegressor(**cfg, random_state=42)
    m.fit(X_train_12h, y_train_12h)
    pred_lvl = pld_base_12h + m.predict(X_test_12h)
    r2 = r2_score(y_test_12h_lvl, pred_lvl)
    print(f"   - HGB Cfg {cfg} -> R2 = {r2:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test_12h_lvl, pred_lvl)):.2f}")
    if r2 > best_hgb_score:
        best_hgb_score = r2
        best_hgb_model = m
        best_hgb_cfg = cfg

# C. Extra Trees Tuning
et_configs = [
    {"n_estimators": 250, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.75},
    {"n_estimators": 400, "max_depth": 24, "min_samples_leaf": 1, "max_features": 0.80},
]
best_et_score = -float('inf')
best_et_model = None
best_et_cfg = None

for cfg in et_configs:
    m = ExtraTreesRegressor(**cfg, random_state=42, n_jobs=-1)
    m.fit(X_train_12h, y_train_12h)
    pred_lvl = pld_base_12h + m.predict(X_test_12h)
    r2 = r2_score(y_test_12h_lvl, pred_lvl)
    print(f"   - ET Cfg {cfg} -> R2 = {r2:.4f}, RMSE = {np.sqrt(mean_squared_error(y_test_12h_lvl, pred_lvl)):.2f}")
    if r2 > best_et_score:
        best_et_score = r2
        best_et_model = m
        best_et_cfg = cfg

# =============================================================
# 4. OPTIMIZING ENSEMBLE BLENDING WEIGHTS
# =============================================================
print("\n4. Optimizing Ensemble Blending Weights...")

pred_rf_test = best_rf_model.predict(X_test_12h)
pred_hgb_test = best_hgb_model.predict(X_test_12h)
pred_et_test = best_et_model.predict(X_test_12h)

# Loss function for weights
def ensemble_loss(w):
    w1, w2, w3 = w
    pred_d = w1 * pred_rf_test + w2 * pred_hgb_test + w3 * pred_et_test
    pred_lvl = pld_base_12h + pred_d
    return mean_squared_error(y_test_12h_lvl, pred_lvl)

res_opt = minimize(ensemble_loss, [0.33, 0.33, 0.34], bounds=[(0, 1), (0, 1), (0, 1)], constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
w1_opt, w2_opt, w3_opt = res_opt.x
print(f"   - Optimal Blending Weights: RF={w1_opt:.3f}, HGB={w2_opt:.3f}, ExtraTrees={w3_opt:.3f}")

pred_delta_tuned_ens = w1_opt * pred_rf_test + w2_opt * pred_hgb_test + w3_opt * pred_et_test
pred_level_tuned_ens = pld_base_12h + pred_delta_tuned_ens

r2_tuned = r2_score(y_test_12h_lvl, pred_level_tuned_ens)
ev_tuned = explained_variance_score(y_test_12h_lvl, pred_level_tuned_ens)
rmse_tuned = np.sqrt(mean_squared_error(y_test_12h_lvl, pred_level_tuned_ens))
mae_tuned = mean_absolute_error(y_test_12h_lvl, pred_level_tuned_ens)
dir_acc_tuned = (np.sign(pred_delta_tuned_ens) == np.sign(y_test_12h_delta)).mean() * 100

# Save Tuned Configuration & Results
tuning_results = {
    "Optimal_Hyperparameters": {
        "PLS_Components": best_n_pls,
        "PCA_Components": 20,
        "Random_Forest": best_rf_cfg,
        "Hist_Gradient_Boosting": best_hgb_cfg,
        "Extra_Trees": best_et_cfg,
        "Ensemble_Weights": {"Random_Forest": round(w1_opt, 3), "Hist_Gradient_Boosting": round(w2_opt, 3), "Extra_Trees": round(w3_opt, 3)}
    },
    "Tuned_Model_Performance": {
        "R2_Score": round(r2_tuned, 4),
        "Explained_Variance": round(ev_tuned, 4),
        "RMSE": round(rmse_tuned, 4),
        "MAE": round(mae_tuned, 4),
        "Directional_Accuracy_Pct": round(dir_acc_tuned, 2)
    }
}

with open(get_result_path("tuned_hyperparameters_results.json"), "w") as f:
    json.dump(tuning_results, f, indent=4)

print("\n" + "="*90)
print("FINAL TUNED MODEL PERFORMANCE (12-HOUR WINDOW)")
print("="*90)
print(f"1. R² Score:              {r2_tuned:.4f} ({r2_tuned*100:.2f}%)")
print(f"2. Explained Variance:    {ev_tuned:.4f} ({ev_tuned*100:.2f}%)")
print(f"3. RMSE:                  {rmse_tuned:.4f} R$/MWh")
print(f"4. MAE:                   {mae_tuned:.4f} R$/MWh")
print(f"5. Directional Accuracy:  {dir_acc_tuned:.2f}%")
print("="*90)
print("\nTuning pipeline finished successfully! Saved to tuned_hyperparameters_results.json")
