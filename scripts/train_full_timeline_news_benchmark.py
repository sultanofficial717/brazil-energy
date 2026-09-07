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

print("="*85)
print("FULL TIMELINE (2018-2026 INCL. 2021-2022) BENCHMARK ON NEWS-ACTIVE SUBSET")
print("="*85)

# 1. LOAD DATASET
print("1. Loading dataset: merged_news_pld_cmo_by_region_date_with_embeddings.csv...")
df = pd.read_csv(get_data_path('merged_news_pld_cmo_by_region_date_with_embeddings.csv'))
df['Date_dt'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# 2. FILTER TO NEWS-ACTIVE SUBSET (Articles >= 1) ACROSS ALL YEARS (2018-2026)
df_news = df[df['Articles'] > 0].copy().reset_index(drop=True)

print(f"   - Total rows in entire 2018-2026 dataset: {len(df)}")
print(f"   - Total News-Active rows (Articles >= 1): {len(df_news)} (100% have valid news text & embeddings)")
print(f"   - Covered Years: {sorted(df_news['Date_dt'].dt.year.unique())}")
print(f"   - Mean Articles/Day: {df_news['Articles'].mean():.2f} (Max: {df_news['Articles'].max()})")

# 3. FEATURE ENGINEERING
print("\n2. Engineering Market Spreads, Volatilities, and Dynamics across Full Timeline...")

# Market spreads & momentum
df_news['intraday_spread'] = df_news['pld_daily_max'] - df_news['pld_daily_min']
df_news['cmo_pld_gap'] = df_news['cmo_weekly_mean'] - df_news['pld_daily_mean']
df_news['cmo_load_spread'] = df_news['cmo_heavy_load'] - df_news['cmo_light_load']

df_news['pld_diff_lag1'] = df_news['pld_daily_mean'] - df_news['pld_lag_1d']
df_news['pld_diff_lag7'] = df_news['pld_daily_mean'] - df_news['pld_lag_7d']
df_news['pld_diff_lag14'] = df_news['pld_daily_mean'] - df_news['pld_lag_14d']
df_news['pld_diff_lag30'] = df_news['pld_daily_mean'] - df_news['pld_lag_30d']
df_news['pld_ratio_lag1'] = df_news['pld_daily_mean'] / (df_news['pld_lag_1d'] + 1e-3)
df_news['pld_ratio_lag7'] = df_news['pld_daily_mean'] / (df_news['pld_lag_7d'] + 1e-3)

# Rolling price volatilities
df_news['pld_rolling_std_7d'] = df_news['pld_daily_mean'].rolling(7, min_periods=1).std().fillna(0)
df_news['pld_rolling_std_14d'] = df_news['pld_daily_mean'].rolling(14, min_periods=1).std().fillna(0)
df_news['pld_rolling_std_30d'] = df_news['pld_daily_mean'].rolling(30, min_periods=1).std().fillna(0)

# News sentiment rolling dynamics
df_news['news_sent_roll3'] = df_news['Avg Sentiment'].rolling(3, min_periods=1).mean()
df_news['news_sent_roll7'] = df_news['Avg Sentiment'].rolling(7, min_periods=1).mean()
df_news['news_imp_roll7'] = df_news['Avg Importance'].rolling(7, min_periods=1).mean()
df_news['news_hydro_roll7'] = df_news['hydro reservoir levels'].rolling(7, min_periods=1).mean()
df_news['news_drought_roll7'] = df_news['Drought'].rolling(7, min_periods=1).mean()
df_news['news_flood_roll7'] = df_news['Flood'].rolling(7, min_periods=1).mean()
df_news['news_articles_roll7'] = df_news['Articles'].rolling(7, min_periods=1).mean()

# Stationary Target: Day-Ahead Price Delta
df_news['target_delta'] = df_news['target_pld_next_day'] - df_news['pld_daily_mean']

# 4. PARSE BGE EMBEDDINGS & COMPUTE PCA & SUPERVISED PLS
print("\n3. Parsing BGE 1024-dim Embeddings & Extracting Supervised/Dense Representations...")
def parse_vector(val, dim=1024):
    if pd.isna(val) or not isinstance(val, str) or val == '':
        return np.zeros(dim, dtype=np.float32)
    try:
        return np.array(json.loads(val), dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)

bge_matrix = np.vstack([parse_vector(x, 1024) for x in df_news['bg gme embedding']])
bge_df = pd.DataFrame(bge_matrix, columns=[f'bge_{i}' for i in range(1024)])

# Chronological Train / Test Split (80% Train, 20% Test)
split_idx = int(len(df_news) * 0.8)
y_train_delta = df_news.iloc[:split_idx]['target_delta']
y_test_delta = df_news.iloc[split_idx:]['target_delta']
y_test_level = df_news.iloc[split_idx:]['target_pld_next_day']
pld_mean_test = df_news.iloc[split_idx:]['pld_daily_mean']

print(f"   - Chronological Split: Train={len(y_train_delta)} rows | Test={len(y_test_delta)} rows")
print(f"   - Full Test Target Variance: {y_test_level.var():.2f}")

# A. Unsupervised PCA (16 components)
pca = PCA(n_components=16, random_state=42)
pca.fit(bge_df.iloc[:split_idx])
pca_cols = [f'emb_pca_{i+1:02d}' for i in range(16)]
pca_df = pd.DataFrame(pca.transform(bge_df), columns=pca_cols)

# B. Supervised PLS (10 price-covariance components)
pls = PLSRegression(n_components=10)
pls.fit(bge_matrix[:split_idx], y_train_delta)
pls_cols = [f'emb_pls_{i+1:02d}' for i in range(10)]
pls_df = pd.DataFrame(pls.transform(bge_matrix), columns=pls_cols)

# Rolling embedding momentum
for col in pca_cols[:6]:
    df_news[f'{col}_roll3'] = pca_df[col].rolling(3, min_periods=1).mean()
    df_news[f'{col}_roll7'] = pca_df[col].rolling(7, min_periods=1).mean()

roll_emb_cols = [f'{col}_roll3' for col in pca_cols[:6]] + [f'{col}_roll7' for col in pca_cols[:6]]

# 5. FEATURE GROUPS DEFINITION
# Group 1: Market Only Features (No news of any kind)
mkt_only_cols = [
    'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
    'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
    'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
    'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load',
    'intraday_spread', 'cmo_pld_gap', 'cmo_load_spread', 'pld_diff_lag1',
    'pld_diff_lag7', 'pld_diff_lag14', 'pld_diff_lag30', 'pld_ratio_lag1',
    'pld_ratio_lag7', 'pld_rolling_std_7d', 'pld_rolling_std_14d', 'pld_rolling_std_30d'
]

# Group 2: News Topic & Sentiment Features (Counts and Scores)
news_topic_cols = [
    'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
    'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
    'transmission constraints', 'renewable generation forecasts',
    'demand forecasts (Carga)', 'risk of future energy shortages',
    'Avg Sentiment', 'Avg Importance', 'news_sent_roll3', 'news_sent_roll7',
    'news_imp_roll7', 'news_hydro_roll7', 'news_drought_roll7', 'news_flood_roll7', 'news_articles_roll7'
]

# Group 3: News Embedding Features (Dense Semantics)
embedding_cols = pca_cols + pls_cols + roll_emb_cols
embedding_df = pd.concat([pca_df, pls_df, df_news[roll_emb_cols]], axis=1)

# Construct Feature Matrices
X_mkt = df_news[mkt_only_cols]
X_mkt_topics = df_news[mkt_only_cols + news_topic_cols]
X_full = pd.concat([df_news[mkt_only_cols + news_topic_cols], embedding_df], axis=1)
X_news_only = pd.concat([df_news[news_topic_cols], embedding_df], axis=1)

# Splits
X_train_mkt, X_test_mkt = X_mkt.iloc[:split_idx], X_mkt.iloc[split_idx:]
X_train_topics, X_test_topics = X_mkt_topics.iloc[:split_idx], X_mkt_topics.iloc[split_idx:]
X_train_full, X_test_full = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
X_train_news_only, X_test_news_only = X_news_only.iloc[:split_idx], X_news_only.iloc[split_idx:]

# 6. MODEL TRAINING & COMPARISON
print("\n4. Training & Evaluating Models on Full 2018-2026 Timeline...")

def evaluate_model(name, train_x, test_x):
    m_rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    m_hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.03, l2_regularization=1.0, random_state=42)
    m_et = ExtraTreesRegressor(n_estimators=250, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    
    m_rf.fit(train_x, y_train_delta)
    m_hgb.fit(train_x, y_train_delta)
    m_et.fit(train_x, y_train_delta)
    
    pred_d = 0.40 * m_rf.predict(test_x) + 0.40 * m_hgb.predict(test_x) + 0.20 * m_et.predict(test_x)
    pred_level = pld_mean_test + pred_d
    
    r2 = r2_score(y_test_level, pred_level)
    ev = explained_variance_score(y_test_level, pred_level)
    rmse = np.sqrt(mean_squared_error(y_test_level, pred_level))
    mae = mean_absolute_error(y_test_level, pred_level)
    dir_acc = (np.sign(pred_d) == np.sign(y_test_delta)).mean() * 100
    
    return {
        "Config": name,
        "R2": round(r2, 4),
        "Explained_Variance": round(ev, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Directional_Accuracy_Pct": round(dir_acc, 2),
        "RF_Model": m_rf
    }

print("   - Training Config 1: Market Only (Without News/Embeddings)...")
res_mkt = evaluate_model("1. Market Only (No News/Embeddings)", X_train_mkt, X_test_mkt)

print("   - Training Config 2: Market + News Topic Counts & Sentiment...")
res_topics = evaluate_model("2. Market + News Topic & Sentiment", X_train_topics, X_test_topics)

print("   - Training Config 3: Full Pipeline (Market + Topics + BGE News Embeddings)...")
res_full = evaluate_model("3. Full Pipeline (+ BGE News Embeddings)", X_train_full, X_test_full)

print("   - Training Config 4: News Alone (Topics + Embeddings Only)...")
res_news_only = evaluate_model("4. News Alone (Topics + Embeddings)", X_train_news_only, X_test_news_only)

# 7. FEATURE IMPORTANCES FROM FULL MODEL
rf_full = res_full["RF_Model"]
imp_series = pd.Series(rf_full.feature_importances_, index=X_train_full.columns).sort_values(ascending=False)

# 8. SAVE REPORT JSON
all_results = {
    "Timeline": "Full 2018-2026 (Including 2021-2022 Crisis)",
    "Dataset": "News-Active Days Only (Articles >= 1)",
    "Total_Rows": len(df_news),
    "Train_Rows": len(y_train_delta),
    "Test_Rows": len(y_test_delta),
    "Comparison": [
        {k: v for k, v in res_mkt.items() if k != "RF_Model"},
        {k: v for k, v in res_topics.items() if k != "RF_Model"},
        {k: v for k, v in res_full.items() if k != "RF_Model"},
        {k: v for k, v in res_news_only.items() if k != "RF_Model"}
    ],
    "Top_20_Feature_Importances": imp_series.head(20).to_dict()
}

with open(get_result_path("full_timeline_news_benchmark_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)

# 9. PRINT COMPARISON TABLE
print("\n" + "="*95)
print("FINAL BENCHMARK COMPARISON ON FULL TIMELINE (2018-2026 INCL. 2021-2022)")
print("="*95)
print(f"{'Configuration':<42} | {'R2 Score':<10} | {'Expl. Var':<10} | {'RMSE (R$)':<10} | {'MAE (R$)':<10} | {'Dir. Acc %':<10}")
print("-" * 105)
for res in all_results["Comparison"]:
    print(f"{res['Config']:<42} | {res['R2']:.4f}{'':<4} | {res['Explained_Variance']:.4f}{'':<4} | {res['RMSE']:<10.4f} | {res['MAE']:<10.4f} | {res['Directional_Accuracy_Pct']:.2f}%")
print("-" * 105)

print("\nTop 15 Feature Importances in Full Pipeline (Full 2018-2026 Timeline):")
for k, v in imp_series.head(15).items():
    print(f"   - {k:<34}: {v:.6f} ({v*100:.2f}%)")

print("\nBenchmark completed successfully! Saved to full_timeline_news_benchmark_results.json")
