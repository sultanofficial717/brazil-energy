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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from scipy.stats import pearsonr, spearmanr

print("="*80)
print("RELATION ANALYSIS: NEWS EMBEDDINGS vs PRICE CHANGES (12-HOUR vs 24-HOUR)")
print("="*80)

# 1. LOAD DATASET
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_clean.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# Retain news-active rows
news_mask = (df['Articles'] > 0).values
df_clean = df[news_mask].copy().reset_index(drop=True)

def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

print("Parsing 1024-dim BGE news embeddings...")
bge_matrix = np.vstack([parse_vector(x, 1024) for x in df_clean['bg gme embedding']])
bge_cols = [f'bge_{i}' for i in range(1024)]
bge_clean = pd.DataFrame(bge_matrix, columns=bge_cols)

# =============================================================
# DATASET 1: 24-HOUR (1-DAY) WINDOW
# =============================================================
print("\n--- Constructing 24-Hour (1-Day) Dataset ---")
df_24h = df_clean.copy()
df_24h['delta_pld_24h'] = df_24h['target_pld_next_day'] - df_24h['pld_daily_mean']
df_24h['pct_delta_24h'] = df_24h['delta_pld_24h'] / (df_24h['pld_daily_mean'] + 1e-3)

hist_mkt_cols_24h = [
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
    'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
    'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
    'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load'
]
news_topic_cols = [
    'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
    'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
    'Avg Sentiment', 'Avg Importance'
]

valid_24h = df_24h[hist_mkt_cols_24h + news_topic_cols + ['delta_pld_24h', 'target_pld_next_day', 'pld_daily_mean']].notnull().all(axis=1)
df_24h = df_24h[valid_24h].reset_index(drop=True)
bge_24h = bge_clean[valid_24h].reset_index(drop=True)

# Chronological split 80% train, 20% test
split_24h = int(len(df_24h) * 0.8)

y_tr_24h_d = df_24h.iloc[:split_24h]['delta_pld_24h']
y_te_24h_d = df_24h.iloc[split_24h:]['delta_pld_24h']
y_te_24h_lvl = df_24h.iloc[split_24h:]['target_pld_next_day']
base_24h_te = df_24h.iloc[split_24h:]['pld_daily_mean']

# Supervised PLS & PCA on 24h embeddings
pls_24h = PLSRegression(n_components=10)
pls_24h.fit(bge_24h.iloc[:split_24h], y_tr_24h_d)
pls_24h_df = pd.DataFrame(pls_24h.transform(bge_24h), columns=[f'pls_{i+1:02d}' for i in range(10)])

pca_24h = PCA(n_components=16, random_state=42)
pca_24h.fit(bge_24h.iloc[:split_24h])
pca_24h_df = pd.DataFrame(pca_24h.transform(bge_24h), columns=[f'pca_{i+1:02d}' for i in range(16)])

X_24h_mkt = df_24h[hist_mkt_cols_24h]
X_24h_full = pd.concat([df_24h[hist_mkt_cols_24h + news_topic_cols], pls_24h_df, pca_24h_df], axis=1)

# =============================================================
# DATASET 2: 12-HOUR WINDOW
# =============================================================
print("\n--- Constructing 12-Hour Dataset ---")
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
df_12h['delta_pld_12h'] = df_12h['target_block_pld'] - df_12h['block_pld']

valid_12h = df_12h.notnull().all(axis=1)
df_12h = df_12h[valid_12h].reset_index(drop=True)
bge_12h = bge_12h[valid_12h].reset_index(drop=True)

split_12h = int(len(df_12h) * 0.8)
y_tr_12h_d = df_12h.iloc[:split_12h]['delta_pld_12h']
y_te_12h_d = df_12h.iloc[split_12h:]['delta_pld_12h']
y_te_12h_lvl = df_12h.iloc[split_12h:]['target_block_pld']
base_12h_te = df_12h.iloc[split_12h:]['block_pld']

pls_12h = PLSRegression(n_components=10)
pls_12h.fit(bge_12h.iloc[:split_12h], y_tr_12h_d)
pls_12h_df = pd.DataFrame(pls_12h.transform(bge_12h), columns=[f'pls_{i+1:02d}' for i in range(10)])

pca_12h = PCA(n_components=16, random_state=42)
pca_12h.fit(bge_12h.iloc[:split_12h])
pca_12h_df = pd.DataFrame(pca_12h.transform(bge_12h), columns=[f'pca_{i+1:02d}' for i in range(16)])

feat_mkt_12h = [
    'block_pld', 'block_cmo', 'intraday_spread', 'block_cmo_gap',
    'block_pld_lag1', 'block_pld_lag2', 'block_diff_lag1', 'block_rolling_std_6',
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'cmo_weekly_mean'
]
X_12h_mkt = df_12h[feat_mkt_12h]
X_12h_full = pd.concat([df_12h[feat_mkt_12h + news_topic_cols], pls_12h_df, pca_12h_df], axis=1)

# =============================================================
# STATISTICAL CORRELATION COMPARISON
# =============================================================
print("\n" + "="*80)
print("STATISTICAL CORRELATIONS WITH PRICE CHANGE (DELTA PLD)")
print("="*80)

# Top PLS component correlation
r_pls1_12h, p_pls1_12h = pearsonr(pls_12h_df['pls_01'].iloc[split_12h:], y_te_12h_d)
r_pls2_12h, _ = pearsonr(pls_12h_df['pls_02'].iloc[split_12h:], y_te_12h_d)
rho_pls1_12h, _ = spearmanr(pls_12h_df['pls_01'].iloc[split_12h:], y_te_12h_d)

r_pls1_24h, p_pls1_24h = pearsonr(pls_24h_df['pls_01'].iloc[split_24h:], y_te_24h_d)
r_pls2_24h, _ = pearsonr(pls_24h_df['pls_02'].iloc[split_24h:], y_te_24h_d)
rho_pls1_24h, _ = spearmanr(pls_24h_df['pls_01'].iloc[split_24h:], y_te_24h_d)

# Safe correlation function
def safe_corr(x, y, method='pearson'):
    if np.all(x == x.iloc[0]) or np.all(y == y.iloc[0]):
        return 0.0, 1.0
    if method == 'spearman':
        res = spearmanr(x, y)
        return float(res.statistic if hasattr(res, 'statistic') else res[0]), float(res.pvalue if hasattr(res, 'pvalue') else res[1])
    else:
        res = pearsonr(x, y)
        return float(res.statistic if hasattr(res, 'statistic') else res[0]), float(res.pvalue if hasattr(res, 'pvalue') else res[1])

# Sentiment correlation with delta
r_sent_12h, _ = safe_corr(df_12h.iloc[split_12h:]['Avg Sentiment'], y_te_12h_d)
r_sent_24h, _ = safe_corr(df_24h.iloc[split_24h:]['Avg Sentiment'], y_te_24h_d)

# Articles volume correlation with price volatility (abs delta)
r_vol_12h, _ = safe_corr(df_12h.iloc[split_12h:]['Articles'], np.abs(y_te_12h_d))
r_vol_24h, _ = safe_corr(df_24h.iloc[split_24h:]['Articles'], np.abs(y_te_24h_d))

# Drought & Hydro correlation with price level
r_drought_12h, _ = safe_corr(df_12h.iloc[split_12h:]['Drought'], y_te_12h_lvl)
r_drought_24h, _ = safe_corr(df_24h.iloc[split_24h:]['Drought'], y_te_24h_lvl)

print(f"{'Metric / Feature':<35} | {'12-Hour Window':<20} | {'24-Hour Window':<20}")
print("-" * 80)
print(f"{'PLS Embedding Comp 1 vs Delta PLD (r)':<35} | {r_pls1_12h:+.4f} (p={p_pls1_12h:.1e})    | {r_pls1_24h:+.4f} (p={p_pls1_24h:.1e})")
print(f"{'PLS Embedding Comp 1 (Spearman rho)':<35} | {rho_pls1_12h:+.4f}               | {rho_pls1_24h:+.4f}")
print(f"{'PLS Embedding Comp 2 vs Delta PLD (r)':<35} | {r_pls2_12h:+.4f}               | {r_pls2_24h:+.4f}")
print(f"{'Avg Sentiment vs Delta PLD (r)':<35} | {r_sent_12h:+.4f}               | {r_sent_24h:+.4f}")
print(f"{'Article Volume vs |Delta PLD| Vol':<35} | {r_vol_12h:+.4f}               | {r_vol_24h:+.4f}")
print(f"{'Drought Mentions vs PLD Level (r)':<35} | {r_drought_12h:+.4f}               | {r_drought_24h:+.4f}")

# =============================================================
# MODEL EVALUATION & FEATURE IMPORTANCE: 12H vs 24H
# =============================================================
def train_and_get_rf(X_tr, X_te, y_tr, y_te, base_te, y_te_lvl):
    rf = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred_d = rf.predict(X_te)
    pred_lvl = base_te.values + pred_d
    r2 = r2_score(y_te_lvl, pred_lvl)
    rmse = np.sqrt(mean_squared_error(y_te_lvl, pred_lvl))
    mae = mean_absolute_error(y_te_lvl, pred_lvl)
    dir_acc = (np.sign(pred_d) == np.sign(y_te)).mean() * 100
    return rf, r2, rmse, mae, dir_acc, pred_d

print("\nFitting models for 12h...")
rf_12h_mkt, r2_12h_mkt, rmse_12h_mkt, mae_12h_mkt, dir_12h_mkt, pred_12h_mkt_d = train_and_get_rf(
    X_12h_mkt.iloc[:split_12h], X_12h_mkt.iloc[split_12h:], y_tr_12h_d, y_te_12h_d, base_12h_te, y_te_12h_lvl
)
rf_12h_full, r2_12h_full, rmse_12h_full, mae_12h_full, dir_12h_full, pred_12h_full_d = train_and_get_rf(
    X_12h_full.iloc[:split_12h], X_12h_full.iloc[split_12h:], y_tr_12h_d, y_te_12h_d, base_12h_te, y_te_12h_lvl
)

print("Fitting models for 24h...")
rf_24h_mkt, r2_24h_mkt, rmse_24h_mkt, mae_24h_mkt, dir_24h_mkt, pred_24h_mkt_d = train_and_get_rf(
    X_24h_mkt.iloc[:split_24h], X_24h_mkt.iloc[split_24h:], y_tr_24h_d, y_te_24h_d, base_24h_te, y_te_24h_lvl
)
rf_24h_full, r2_24h_full, rmse_24h_full, mae_24h_full, dir_24h_full, pred_24h_full_d = train_and_get_rf(
    X_24h_full.iloc[:split_24h], X_24h_full.iloc[split_24h:], y_tr_24h_d, y_te_24h_d, base_24h_te, y_te_24h_lvl
)

# Feature Importance aggregation
def group_feature_importances(rf_model, feat_names):
    fi = rf_model.feature_importances_
    mkt_imp = 0.0
    topic_imp = 0.0
    emb_imp = 0.0
    for name, imp in zip(feat_names, fi):
        if name.startswith('pls_') or name.startswith('pca_'):
            emb_imp += imp
        elif name in news_topic_cols:
            topic_imp += imp
        else:
            mkt_imp += imp
    return mkt_imp * 100, topic_imp * 100, emb_imp * 100

mkt_imp_12h, top_imp_12h, emb_imp_12h = group_feature_importances(rf_12h_full, X_12h_full.columns)
mkt_imp_24h, top_imp_24h, emb_imp_24h = group_feature_importances(rf_24h_full, X_24h_full.columns)

# Save results JSON
results_dict = {
    "12_Hour": {
        "Correlation_PLS1_Delta": round(float(r_pls1_12h), 4),
        "Correlation_Sentiment_Delta": round(float(r_sent_12h), 4),
        "Correlation_Volume_Volatility": round(float(r_vol_12h), 4),
        "R2_Market": round(float(r2_12h_mkt), 4),
        "R2_Full": round(float(r2_12h_full), 4),
        "R2_Lift_Pct": round(float((r2_12h_full - r2_12h_mkt)*100), 2),
        "RMSE_Market": round(float(rmse_12h_mkt), 2),
        "RMSE_Full": round(float(rmse_12h_full), 2),
        "MAE_Market": round(float(mae_12h_mkt), 2),
        "MAE_Full": round(float(mae_12h_full), 2),
        "Directional_Accuracy_Pct": round(float(dir_12h_full), 2),
        "Feature_Importance": {
            "Market": round(float(mkt_imp_12h), 1),
            "News_Topics": round(float(top_imp_12h), 1),
            "News_Embeddings": round(float(emb_imp_12h), 1)
        }
    },
    "24_Hour": {
        "Correlation_PLS1_Delta": round(float(r_pls1_24h), 4),
        "Correlation_Sentiment_Delta": round(float(r_sent_24h), 4),
        "Correlation_Volume_Volatility": round(float(r_vol_24h), 4),
        "R2_Market": round(float(r2_24h_mkt), 4),
        "R2_Full": round(float(r2_24h_full), 4),
        "R2_Lift_Pct": round(float((r2_24h_full - r2_24h_mkt)*100), 2),
        "RMSE_Market": round(float(rmse_24h_mkt), 2),
        "RMSE_Full": round(float(rmse_24h_full), 2),
        "MAE_Market": round(float(mae_24h_mkt), 2),
        "MAE_Full": round(float(mae_24h_full), 2),
        "Directional_Accuracy_Pct": round(float(dir_24h_full), 2),
        "Feature_Importance": {
            "Market": round(float(mkt_imp_24h), 1),
            "News_Topics": round(float(top_imp_24h), 1),
            "News_Embeddings": round(float(emb_imp_24h), 1)
        }
    }
}

with open(get_result_path("embeddings_relation_12h_vs_24h.json"), "w") as f:
    json.dump(results_dict, f, indent=4)
print("Saved embeddings_relation_12h_vs_24h.json")

# =============================================================
# GENERATE COMPARATIVE VISUALIZATION: 12H vs 24H
# =============================================================
print("Generating comparative figure...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: PLS Component 1 vs Delta PLD (12h vs 24h)
ax1 = axes[0, 0]
test_pls1_12h = pls_12h_df['pls_01'].iloc[split_12h:]
ax1.scatter(test_pls1_12h, y_te_12h_d, alpha=0.25, color='#0284c7', s=16, label=f'12h Δ PLD (r = {r_pls1_12h:+.3f})')
z1 = np.polyfit(test_pls1_12h, y_te_12h_d, 1)
p1 = np.poly1d(z1)
ax1.plot(np.sort(test_pls1_12h), p1(np.sort(test_pls1_12h)), color='#0369a1', linewidth=2.2, label='12h Linear Trend')
ax1.set_title("12-Hour: News Embedding (PLS Comp 1) vs Price Change", fontsize=12, fontweight='bold')
ax1.set_xlabel("Latent News Embedding Signal (Supervised PLS 1)", fontsize=11)
ax1.set_ylabel("12-Hour Δ PLD (R$/MWh)", fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: PLS Component 1 vs Delta PLD 24h
ax2 = axes[0, 1]
test_pls1_24h = pls_24h_df['pls_01'].iloc[split_24h:]
ax2.scatter(test_pls1_24h, y_te_24h_d, alpha=0.3, color='#10b981', s=20, label=f'24h Δ PLD (r = {r_pls1_24h:+.3f})')
z2 = np.polyfit(test_pls1_24h, y_te_24h_d, 1)
p2 = np.poly1d(z2)
ax2.plot(np.sort(test_pls1_24h), p2(np.sort(test_pls1_24h)), color='#047857', linewidth=2.2, label='24h Linear Trend')
ax2.set_title("24-Hour (1-Day): News Embedding (PLS Comp 1) vs Price Change", fontsize=12, fontweight='bold')
ax2.set_xlabel("Latent News Embedding Signal (Supervised PLS 1)", fontsize=11)
ax2.set_ylabel("24-Hour Δ PLD (R$/MWh)", fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Feature Importance Breakdown (12h vs 24h)
ax3 = axes[1, 0]
categories = ['12-Hour Horizon', '24-Hour Horizon']
mkt_vals = [mkt_imp_12h, mkt_imp_24h]
top_vals = [top_imp_12h, top_imp_24h]
emb_vals = [emb_imp_12h, emb_imp_24h]

x = np.arange(len(categories))
width = 0.55
ax3.bar(x, mkt_vals, width, label='Historical Market Features', color='#64748b', alpha=0.9)
ax3.bar(x, top_vals, width, bottom=mkt_vals, label='News Sentiment & Topics', color='#f59e0b', alpha=0.9)
ax3.bar(x, emb_vals, width, bottom=np.array(mkt_vals)+np.array(top_vals), label='Dense News Embeddings (PLS+PCA)', color='#3b82f6', alpha=0.9)

for i in range(len(categories)):
    total = mkt_vals[i] + top_vals[i] + emb_vals[i]
    ax3.text(i, mkt_vals[i]/2, f"{mkt_vals[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax3.text(i, mkt_vals[i] + top_vals[i]/2, f"{top_vals[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax3.text(i, mkt_vals[i] + top_vals[i] + emb_vals[i]/2, f"{emb_vals[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=11, fontweight='semibold')
ax3.set_title("Relative Feature Importance: Market vs News Embeddings", fontsize=12, fontweight='bold')
ax3.set_ylabel("Feature Importance Share (%)", fontsize=11)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.5, axis='y')

# Subplot 4: Model Performance Metrics (R2 & MAE Comparison)
ax4 = axes[1, 1]
labels = ['12h Market', '12h Full (Embeddings)', '24h Market', '24h Full (Embeddings)']
r2_scores = [r2_12h_mkt, r2_12h_full, r2_24h_mkt, r2_24h_full]
maes = [mae_12h_mkt, mae_12h_full, mae_24h_mkt, mae_24h_full]
colors = ['#94a3b8', '#0284c7', '#cbd5e1', '#10b981']

bars = ax4.bar(labels, r2_scores, color=colors, width=0.55, edgecolor='black', linewidth=0.5)
ax4.set_ylim(0.5, 1.0)
ax4.set_title("Predictive Accuracy (R² Score) Lift from News Embeddings", fontsize=12, fontweight='bold')
ax4.set_ylabel("R² Score (Test Horizon)", fontsize=11)
ax4.grid(True, linestyle='--', alpha=0.5, axis='y')

for bar, r2, mae in zip(bars, r2_scores, maes):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"R²={r2:.4f}\nMAE={mae:.1f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
plt.savefig(get_figure_path('fig_embeddings_relation_12h_vs_24h.png'), dpi=200)
plt.close()
print("Saved comparison chart to fig_embeddings_relation_12h_vs_24h.png")
print("ALL DONE!")
