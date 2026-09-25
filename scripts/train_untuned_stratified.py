#!/usr/bin/env python3
"""
scripts/train_untuned_stratified.py
===================================
Evaluates the UN-TUNED baseline pipelines on the stratified dataset split:
1. Un-tuned Random Forest (n_estimators=150, max_depth=15, min_samples_leaf=2, max_features=1.0, 32 PCA components)
2. Un-tuned Multi-Model Ensemble (40% RF + 40% HGB + 20% ET, 10 PLS + 16 PCA components)
3. Evaluates:
   - Without News (Historical Market Only)
   - With News Topics & Sentiment
   - With Full News Embeddings
   - News Embeddings Alone
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from paths import get_data_path, get_result_path

def main():
    print("="*85)
    print("UN-TUNED PIPELINES EVALUATED ON STRATIFIED DATASET SPLIT")
    print("="*85)

    # 1. Load dataset
    csv_path = Path(get_data_path('merged_news_pld_cmo_by_region_date_clean.csv'))
    if not csv_path.exists():
        csv_path = Path('data/merged_news_pld_cmo_by_region_date_with_embeddings_backup.csv')
    print(f"Loading data from: {csv_path}...")
    df = pd.read_csv(csv_path)

    target_col = 'target_pld_next_day'
    df = df[df[target_col].notnull()].reset_index(drop=True)

    # 2. Features
    hist_market_cols = [
        'pld_daily_mean', 'pld_daily_min', 'pld_daily_max', 'pld_hour_count',
        'pld_lag_1d', 'pld_lag_7d', 'pld_lag_14d', 'pld_lag_30d',
        'pld_rolling_mean_7d', 'pld_rolling_mean_14d', 'pld_rolling_mean_30d',
        'cmo_weekly_mean', 'cmo_light_load', 'cmo_medium_load', 'cmo_heavy_load'
    ]

    news_topic_cols = [
        'Articles', 'Flood', 'Drought', 'Curtailment', 'El nino', 'La nina',
        'hydro reservoir levels', 'future rainfall uncertainty', 'thermal fuel costs',
        'transmission constraints', 'renewable generation forecasts',
        'demand forecasts (Carga)', 'risk of future energy shortages',
        'Avg Sentiment', 'Avg Importance'
    ]

    topic_names = [
        'future rainfall uncertainty', 'Flood', 'El nino', 'Drought',
        'thermal fuel costs', 'La nina', 'hydro reservoir levels',
        'transmission constraints', 'Curtailment', 'renewable generation forecasts',
        'demand forecasts (Carga)', 'risk of future energy shortages'
    ]

    # Assign dominant topic for topic stratification
    def assign_topic_class(row):
        if row['Articles'] == 0:
            return 'No News'
        best_topic = 'General Energy'
        max_val = 0
        for t in topic_names:
            if row[t] > max_val:
                max_val = row[t]
                best_topic = t
        return best_topic

    df['topic_class'] = df.apply(assign_topic_class, axis=1)

    # Group rare classes (< 10 samples)
    topic_counts = df['topic_class'].value_counts()
    rare_topics = topic_counts[topic_counts < 10].index.tolist()
    if rare_topics:
        df['stratify_label'] = df['topic_class'].replace(rare_topics, 'Other Topic')
    else:
        df['stratify_label'] = df['topic_class']

    # 3. Parse BGE Embeddings
    print("Parsing 1024-d BGE text embeddings...")
    def parse_vector(val, dim=1024):
        if pd.isna(val) or not isinstance(val, str) or val == '':
            return np.zeros(dim, dtype=np.float32)
        try:
            return np.array(json.loads(val), dtype=np.float32)
        except Exception:
            return np.zeros(dim, dtype=np.float32)

    bge_matrix = np.vstack([parse_vector(x, 1024) for x in df['bg gme embedding']])
    bge_df = pd.DataFrame(bge_matrix, columns=[f'bge_{i}' for i in range(1024)])

    # 4. Stratified Split (80% Train, 20% Test)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=df['stratify_label']
    )

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    y_train = df_train[target_col].values
    y_test = df_test[target_col].values

    print(f"Stratified Split: Train={len(df_train)} rows (80%) | Test={len(df_test)} rows (20%)")

    # 5. Extract Un-tuned Components (strictly on Train)
    # A. Un-tuned PCA: 32 components (as in train_rf_embeddings_pipeline.py)
    print("\nFitting Un-tuned PCA (32 components)...")
    pca_32 = PCA(n_components=32, random_state=42)
    pca_32.fit(bge_df.iloc[train_idx])
    train_pca_32 = pd.DataFrame(pca_32.transform(bge_df.iloc[train_idx]), columns=[f'news_emb_pca_{i+1:02d}' for i in range(32)])
    test_pca_32 = pd.DataFrame(pca_32.transform(bge_df.iloc[test_idx]), columns=[f'news_emb_pca_{i+1:02d}' for i in range(32)])

    # B. Un-tuned PLS (10 components) + PCA (16 components) (as in train_windows_benchmark.py)
    print("Fitting Un-tuned PLS (10 components) & PCA (16 components)...")
    pls_10 = PLSRegression(n_components=10)
    pls_10.fit(bge_df.iloc[train_idx], y_train)
    train_pls_10 = pd.DataFrame(pls_10.transform(bge_df.iloc[train_idx]), columns=[f'emb_pls_{i+1:02d}' for i in range(10)])
    test_pls_10 = pd.DataFrame(pls_10.transform(bge_df.iloc[test_idx]), columns=[f'emb_pls_{i+1:02d}' for i in range(10)])

    pca_16 = PCA(n_components=16, random_state=42)
    pca_16.fit(bge_df.iloc[train_idx])
    train_pca_16 = pd.DataFrame(pca_16.transform(bge_df.iloc[train_idx]), columns=[f'emb_pca_{i+1:02d}' for i in range(16)])
    test_pca_16 = pd.DataFrame(pca_16.transform(bge_df.iloc[test_idx]), columns=[f'emb_pca_{i+1:02d}' for i in range(16)])

    # Feature sets
    X_train_mkt = df_train[hist_market_cols].copy()
    X_test_mkt = df_test[hist_market_cols].copy()

    X_train_topics = df_train[hist_market_cols + news_topic_cols].copy()
    X_test_topics = df_test[hist_market_cols + news_topic_cols].copy()

    # Full features for RF (with 32 PCA components)
    X_train_rf_full = pd.concat([X_train_topics, train_pca_32], axis=1)
    X_test_rf_full = pd.concat([X_test_topics, test_pca_32], axis=1)

    # Full features for Ensemble (with 10 PLS + 16 PCA)
    X_train_ens_full = pd.concat([X_train_topics, train_pls_10, train_pca_16], axis=1)
    X_test_ens_full = pd.concat([X_test_topics, test_pls_10, test_pca_16], axis=1)

    def calc_metrics(preds):
        r2 = r2_score(y_test, preds)
        ev = explained_variance_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.clip(np.abs(y_test), 1.0, None))) * 100
        return {
            "R2": round(r2, 4),
            "Explained_Variance": round(ev, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 2)
        }

    # =========================================================================
    # PIPELINE 1: UN-TUNED RANDOM FOREST (from scripts/train_rf_embeddings_pipeline.py)
    # n_estimators=150, max_depth=15, min_samples_leaf=2, max_features=None
    # =========================================================================
    print("\n--- Running Un-tuned Random Forest Pipeline (150 trees, max_depth=15) ---")
    rf_untuned = lambda: RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)

    # 1. Market Only
    rf_mkt = rf_untuned().fit(X_train_mkt, y_train)
    res_rf_mkt = calc_metrics(rf_mkt.predict(X_test_mkt))

    # 2. Market + Topics
    rf_topics = rf_untuned().fit(X_train_topics, y_train)
    res_rf_topics = calc_metrics(rf_topics.predict(X_test_topics))

    # 3. Full Pipeline (+ 32 PCA Embeddings)
    rf_full = rf_untuned().fit(X_train_rf_full, y_train)
    res_rf_full = calc_metrics(rf_full.predict(X_test_rf_full))

    # 4. News Embeddings Alone
    rf_emb = rf_untuned().fit(train_pca_32, y_train)
    res_rf_emb = calc_metrics(rf_emb.predict(test_pca_32))

    # =========================================================================
    # PIPELINE 2: UN-TUNED ENSEMBLE (from scripts/train_windows_benchmark.py)
    # Blend: 40% RF (300 trees, depth 20) + 40% HGB (300 iter, depth 8) + 20% ET (250 trees, depth 20)
    # =========================================================================
    print("--- Running Un-tuned Multi-Model Ensemble (40% RF + 40% HGB + 20% ET) ---")
    m_rf_ens = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)
    m_hgb_ens = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.03, l2_regularization=1.0, random_state=42)
    m_et_ens = ExtraTreesRegressor(n_estimators=250, max_depth=20, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1)

    # Ensemble on Market Only
    m_rf_ens.fit(X_train_mkt, y_train)
    m_hgb_ens.fit(X_train_mkt, y_train)
    m_et_ens.fit(X_train_mkt, y_train)
    pred_ens_mkt = 0.40 * m_rf_ens.predict(X_test_mkt) + 0.40 * m_hgb_ens.predict(X_test_mkt) + 0.20 * m_et_ens.predict(X_test_mkt)
    res_ens_mkt = calc_metrics(pred_ens_mkt)

    # Ensemble on Full Pipeline
    m_rf_ens.fit(X_train_ens_full, y_train)
    m_hgb_ens.fit(X_train_ens_full, y_train)
    m_et_ens.fit(X_train_ens_full, y_train)
    pred_ens_full = 0.40 * m_rf_ens.predict(X_test_ens_full) + 0.40 * m_hgb_ens.predict(X_test_ens_full) + 0.20 * m_et_ens.predict(X_test_ens_full)
    res_ens_full = calc_metrics(pred_ens_full)

    print("\n" + "="*85)
    print("FINAL RESULTS: UN-TUNED PIPELINES ON STRATIFIED SPLIT")
    print("="*85)

    results_table = [
        {
            "Pipeline / Model": "Un-tuned RF: Without News (Market Baseline)",
            "R2": res_rf_mkt["R2"],
            "Explained Variance": res_rf_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_mkt["RMSE"],
            "MAE (R$/MWh)": res_rf_mkt["MAE"],
            "MAPE (%)": res_rf_mkt["MAPE"]
        },
        {
            "Pipeline / Model": "Un-tuned RF: With News Topics & Sentiment",
            "R2": res_rf_topics["R2"],
            "Explained Variance": res_rf_topics["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_topics["RMSE"],
            "MAE (R$/MWh)": res_rf_topics["MAE"],
            "MAPE (%)": res_rf_topics["MAPE"]
        },
        {
            "Pipeline / Model": "Un-tuned RF: With Full Embeddings (32 PCA)",
            "R2": res_rf_full["R2"],
            "Explained Variance": res_rf_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_full["RMSE"],
            "MAE (R$/MWh)": res_rf_full["MAE"],
            "MAPE (%)": res_rf_full["MAPE"]
        },
        {
            "Pipeline / Model": "Un-tuned RF: News Embeddings Alone (Text Only)",
            "R2": res_rf_emb["R2"],
            "Explained Variance": res_rf_emb["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_emb["RMSE"],
            "MAE (R$/MWh)": res_rf_emb["MAE"],
            "MAPE (%)": res_rf_emb["MAPE"]
        },
        {
            "Pipeline / Model": "Un-tuned Ensemble: Without News (Market Baseline)",
            "R2": res_ens_mkt["R2"],
            "Explained Variance": res_ens_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_ens_mkt["RMSE"],
            "MAE (R$/MWh)": res_ens_mkt["MAE"],
            "MAPE (%)": res_ens_mkt["MAPE"]
        },
        {
            "Pipeline / Model": "Un-tuned Ensemble: With Full Embeddings (PLS+PCA)",
            "R2": res_ens_full["R2"],
            "Explained Variance": res_ens_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_ens_full["RMSE"],
            "MAE (R$/MWh)": res_ens_full["MAE"],
            "MAPE (%)": res_ens_full["MAPE"]
        }
    ]

    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

    # Feature Importance of Un-tuned RF Full
    imp_rf = pd.Series(rf_full.feature_importances_, index=X_train_rf_full.columns).sort_values(ascending=False)
    print("\nTop 15 Most Important Features in Un-tuned RF Full Model:")
    for feat, val in imp_rf.head(15).items():
        print(f"   - {feat:<30}: {val:.4f} ({val*100:.2f}%)")

    # Save to JSON
    out_path = Path("results/untuned_stratified_split_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results_table,
            "top_features": imp_rf.head(15).to_dict()
        }, f, indent=4)
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
