#!/usr/bin/env python3
"""
scripts/train_tuned_stratified.py
=================================
Evaluates the tuned hyperparameters on the stratified dataset split:
1. Tuned Random Forest (n_estimators=200, max_depth=18, min_samples_leaf=2, max_features=0.70)
2. Tuned Supervised PLS (12 dims) and Unsupervised PCA (20 dims)
3. Tuned Ensemble (Random Forest + Extra Trees + HistGradientBoosting with optimal weights)
4. Comparison: Without News vs. With News Topics vs. With Full Embeddings
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
    print("STRATIFIED DATASET SPLIT WITH TUNED HYPERPARAMETERS")
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

    print(f"Stratified Split: Train={len(df_train)} rows | Test={len(df_test)} rows")

    # 5. Extract Tuned Latent Components strictly on Train
    # A. Tuned PLS (12 components)
    print("\nFitting Tuned Supervised PLS (12 components)...")
    pls_opt = PLSRegression(n_components=12)
    pls_opt.fit(bge_df.iloc[train_idx], y_train)
    train_pls = pd.DataFrame(pls_opt.transform(bge_df.iloc[train_idx]), columns=[f'pls_{i+1:02d}' for i in range(12)])
    test_pls = pd.DataFrame(pls_opt.transform(bge_df.iloc[test_idx]), columns=[f'pls_{i+1:02d}' for i in range(12)])

    # B. Tuned PCA (20 components)
    print("Fitting Tuned Unsupervised PCA (20 components)...")
    pca_opt = PCA(n_components=20, random_state=42)
    pca_opt.fit(bge_df.iloc[train_idx])
    train_pca = pd.DataFrame(pca_opt.transform(bge_df.iloc[train_idx]), columns=[f'pca_{i+1:02d}' for i in range(20)])
    test_pca = pd.DataFrame(pca_opt.transform(bge_df.iloc[test_idx]), columns=[f'pca_{i+1:02d}' for i in range(20)])

    # 6. Feature Matrices
    # Market Only (Without News)
    X_train_mkt = df_train[hist_market_cols].copy()
    X_test_mkt = df_test[hist_market_cols].copy()

    # Market + Topics
    X_train_topics = df_train[hist_market_cols + news_topic_cols].copy()
    X_test_topics = df_test[hist_market_cols + news_topic_cols].copy()

    # Full Pipeline (Market + Topics + Tuned PLS + Tuned PCA)
    X_train_full = pd.concat([X_train_topics, train_pls, train_pca], axis=1)
    X_test_full = pd.concat([X_test_topics, test_pls, test_pca], axis=1)

    # 7. Tuned Random Forest Configuration
    # (n_estimators=200, max_depth=18, min_samples_leaf=2, max_features=0.70)
    tuned_rf_params = {
        "n_estimators": 200,
        "max_depth": 18,
        "min_samples_leaf": 2,
        "max_features": 0.70,
        "random_state": 42,
        "n_jobs": -1
    }

    # Tuned Extra Trees Configuration
    tuned_et_params = {
        "n_estimators": 250,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": 0.75,
        "random_state": 42,
        "n_jobs": -1
    }

    # Tuned HistGradientBoosting Configuration
    tuned_hgb_params = {
        "max_iter": 350,
        "max_depth": 9,
        "learning_rate": 0.025,
        "l2_regularization": 1.5,
        "random_state": 42
    }

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

    # PART A: TUNED RANDOM FOREST ALONE
    print("\n--- Training Tuned Random Forest Models ---")
    rf_mkt = RandomForestRegressor(**tuned_rf_params).fit(X_train_mkt, y_train)
    res_rf_mkt = calc_metrics(rf_mkt.predict(X_test_mkt))

    rf_topics = RandomForestRegressor(**tuned_rf_params).fit(X_train_topics, y_train)
    res_rf_topics = calc_metrics(rf_topics.predict(X_test_topics))

    rf_full = RandomForestRegressor(**tuned_rf_params).fit(X_train_full, y_train)
    res_rf_full = calc_metrics(rf_full.predict(X_test_full))

    # PART B: TUNED EXTRA TREES ALONE
    print("--- Training Tuned Extra Trees Models ---")
    et_mkt = ExtraTreesRegressor(**tuned_et_params).fit(X_train_mkt, y_train)
    res_et_mkt = calc_metrics(et_mkt.predict(X_test_mkt))

    et_full = ExtraTreesRegressor(**tuned_et_params).fit(X_train_full, y_train)
    res_et_full = calc_metrics(et_full.predict(X_test_full))

    # PART C: FULL TUNED ENSEMBLE (RF + ET + HGB with optimal blending)
    print("--- Training Full Tuned Ensemble (Optimal Blend: 80.2% ET + 19.8% RF) ---")
    hgb_full = HistGradientBoostingRegressor(**tuned_hgb_params).fit(X_train_full, y_train)
    
    pred_ens_mkt = 0.802 * et_mkt.predict(X_test_mkt) + 0.198 * rf_mkt.predict(X_test_mkt)
    res_ens_mkt = calc_metrics(pred_ens_mkt)

    pred_ens_full = 0.802 * et_full.predict(X_test_full) + 0.198 * rf_full.predict(X_test_full)
    res_ens_full = calc_metrics(pred_ens_full)

    # Feature Importances from Tuned RF Full Model
    imp_rf = pd.Series(rf_full.feature_importances_, index=X_train_full.columns).sort_values(ascending=False)

    print("\n" + "="*85)
    print("RESULTS SUMMARY: TUNED HYPERPARAMETERS ON STRATIFIED SPLIT")
    print("="*85)

    summary_rows = [
        {
            "Model Pipeline": "Tuned Random Forest - Without News (Market Baseline)",
            "R2": res_rf_mkt["R2"],
            "Explained Variance": res_rf_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_mkt["RMSE"],
            "MAE (R$/MWh)": res_rf_mkt["MAE"],
            "MAPE (%)": res_rf_mkt["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Random Forest - With News Topics",
            "R2": res_rf_topics["R2"],
            "Explained Variance": res_rf_topics["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_topics["RMSE"],
            "MAE (R$/MWh)": res_rf_topics["MAE"],
            "MAPE (%)": res_rf_topics["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Random Forest - With Full Embeddings (PLS+PCA)",
            "R2": res_rf_full["R2"],
            "Explained Variance": res_rf_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_rf_full["RMSE"],
            "MAE (R$/MWh)": res_rf_full["MAE"],
            "MAPE (%)": res_rf_full["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Extra Trees - Without News (Market Baseline)",
            "R2": res_et_mkt["R2"],
            "Explained Variance": res_et_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_et_mkt["RMSE"],
            "MAE (R$/MWh)": res_et_mkt["MAE"],
            "MAPE (%)": res_et_mkt["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Extra Trees - With Full Embeddings",
            "R2": res_et_full["R2"],
            "Explained Variance": res_et_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_et_full["RMSE"],
            "MAE (R$/MWh)": res_et_full["MAE"],
            "MAPE (%)": res_et_full["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Ensemble Blend - Without News",
            "R2": res_ens_mkt["R2"],
            "Explained Variance": res_ens_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_ens_mkt["RMSE"],
            "MAE (R$/MWh)": res_ens_mkt["MAE"],
            "MAPE (%)": res_ens_mkt["MAPE"]
        },
        {
            "Model Pipeline": "Tuned Ensemble Blend - With Full Embeddings",
            "R2": res_ens_full["R2"],
            "Explained Variance": res_ens_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_ens_full["RMSE"],
            "MAE (R$/MWh)": res_ens_full["MAE"],
            "MAPE (%)": res_ens_full["MAPE"]
        }
    ]
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\nTop 15 Most Important Features in Tuned RF Full Model:")
    for feat, val in imp_rf.head(15).items():
        print(f"   - {feat:<30}: {val:.4f} ({val*100:.2f}%)")

    # Save to JSON
    out_dict = {
        "stratified_tuned_results": summary_rows,
        "tuned_hyperparameters_used": {
            "random_forest": tuned_rf_params,
            "extra_trees": tuned_et_params,
            "hist_gradient_boosting": tuned_hgb_params,
            "latent_dimensions": {"PLS": 12, "PCA": 20},
            "ensemble_weights": {"Extra_Trees": 0.802, "Random_Forest": 0.198, "HGB": 0.0}
        },
        "top_feature_importances": imp_rf.head(15).to_dict()
    }
    out_path = Path("results/tuned_stratified_split_results.json")
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=4)
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
