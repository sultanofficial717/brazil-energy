#!/usr/bin/env python3
"""
scripts/train_rf_stratified.py
==============================
Trains Random Forest models on stratified train/test splits:
1. Stratified by Topic (ensures balanced topic distribution across train and test).
2. Evaluates performance With News and Without News.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from paths import get_data_path, get_result_path

def main():
    print("="*80)
    print("RANDOM FOREST TRAINING WITH STRATIFIED TRAIN/TEST SPLIT")
    print("="*80)

    # 1. Load dataset
    data_path = Path(get_data_path('merged_news_pld_cmo_by_region_date_with_embeddings.csv'))
    if not data_path.exists():
        data_path = Path('data/merged_news_pld_cmo_by_region_date_with_embeddings_backup.csv')
    print(f"Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)

    # Valid target filter
    target_col = 'target_pld_next_day'
    df = df[df[target_col].notnull()].reset_index(drop=True)

    # 2. Define Features
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

    # Group rare classes (< 10 samples) into 'Other Topic' to allow stratified splitting
    topic_counts = df['topic_class'].value_counts()
    rare_topics = topic_counts[topic_counts < 10].index.tolist()
    if rare_topics:
        df['stratify_label'] = df['topic_class'].replace(rare_topics, 'Other Topic')
    else:
        df['stratify_label'] = df['topic_class']

    print(f"\nTotal Dataset Rows: {len(df)}")
    print(f"Stratification Classes (Dominant Topic Distribution):")
    for k, v in df['stratify_label'].value_counts().items():
        print(f"   - {k:<32}: {v:5d} ({v/len(df)*100:5.2f}%)")

    # 3. Parse BGE Embeddings
    print("\nParsing 1024-d BGE text embeddings...")
    def parse_vector(val, dim=1024):
        if pd.isna(val) or not isinstance(val, str) or val == '':
            return np.zeros(dim, dtype=np.float32)
        try:
            return np.array(json.loads(val), dtype=np.float32)
        except Exception:
            return np.zeros(dim, dtype=np.float32)

    bge_matrix = np.vstack([parse_vector(x, 1024) for x in df['bg gme embedding']])
    bge_df = pd.DataFrame(bge_matrix, columns=[f'bge_{i}' for i in range(1024)])

    # 4. Perform Stratified Train/Test Split (80% Train, 20% Test)
    print("\nExecuting Stratified Train/Test Split (80% Train, 20% Test)...")
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

    print(f"   - Training samples: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"   - Testing samples:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

    # Verify Topic Proportion Matching
    print("\nVerification of Topic Stratification Balance (% in Train vs % in Test):")
    balance_records = []
    for cls in df['stratify_label'].unique():
        tr_cnt = (df_train['stratify_label'] == cls).sum()
        te_cnt = (df_test['stratify_label'] == cls).sum()
        tr_pct = tr_cnt / len(df_train) * 100
        te_pct = te_cnt / len(df_test) * 100
        balance_records.append({
            'Topic Class': cls,
            'Train Count': tr_cnt,
            'Train %': round(tr_pct, 2),
            'Test Count': te_cnt,
            'Test %': round(te_pct, 2),
            'Diff (%)': round(abs(tr_pct - te_pct), 3)
        })
    df_balance = pd.DataFrame(balance_records).sort_values('Train Count', ascending=False)
    print(df_balance.to_string(index=False))

    # 5. Fit PCA strictly on Training Embeddings
    print("\nApplying PCA (32 components) fitted strictly on Training split...")
    pca = PCA(n_components=32, random_state=42)
    pca.fit(bge_df.iloc[train_idx])

    pca_cols = [f'news_emb_pca_{i+1:02d}' for i in range(32)]
    train_pca = pd.DataFrame(pca.transform(bge_df.iloc[train_idx]), columns=pca_cols)
    test_pca = pd.DataFrame(pca.transform(bge_df.iloc[test_idx]), columns=pca_cols)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"   - 32 PCA Components explain {var_exp:.2f}% of text semantic variance.")

    # 6. Feature Matrices
    # Without News (Historical Market Only)
    X_train_mkt = df_train[hist_market_cols].copy()
    X_test_mkt = df_test[hist_market_cols].copy()

    # With News Topics (Market + Topics/Sentiment)
    X_train_topics = df_train[hist_market_cols + news_topic_cols].copy()
    X_test_topics = df_test[hist_market_cols + news_topic_cols].copy()

    # With News Full (Market + Topics + 32-d PCA Embeddings)
    X_train_full = pd.concat([X_train_topics, train_pca], axis=1)
    X_test_full = pd.concat([X_test_topics, test_pca], axis=1)

    # 7. Model Training & Evaluation
    def evaluate_model(name, X_tr, X_te):
        print(f"\nTraining Random Forest: {name}...")
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_train)
        preds = rf.predict(X_te)

        r2 = r2_score(y_test, preds)
        ev = explained_variance_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.clip(np.abs(y_test), 1.0, None))) * 100

        print(f"   >>> {name} Results:")
        print(f"       R2 Score:           {r2:.4f} ({r2*100:.2f}%)")
        print(f"       Explained Variance: {ev:.4f} ({ev*100:.2f}%)")
        print(f"       RMSE (R$/MWh):      {rmse:.4f}")
        print(f"       MAE (R$/MWh):       {mae:.4f}")
        print(f"       MAPE (%):           {mape:.2f}%")

        return {
            "R2": round(r2, 4),
            "Explained_Variance": round(ev, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 2),
            "Model": rf
        }

    res_mkt = evaluate_model("WITHOUT NEWS (Market Baseline)", X_train_mkt, X_test_mkt)
    res_topics = evaluate_model("WITH NEWS (Market + Topic Counts)", X_train_topics, X_test_topics)
    res_full = evaluate_model("WITH NEWS (Full Pipeline: Market + Topics + PCA Embeddings)", X_train_full, X_test_full)

    # 8. Comparison Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON (STRATIFIED SPLIT)")
    print("="*80)
    summary_data = [
        {
            "Configuration": "WITHOUT NEWS (Market Only)",
            "R2": res_mkt["R2"],
            "Explained Variance": res_mkt["Explained_Variance"],
            "RMSE (R$/MWh)": res_mkt["RMSE"],
            "MAE (R$/MWh)": res_mkt["MAE"],
            "MAPE (%)": res_mkt["MAPE"],
            "R2 Lift": "Baseline",
            "RMSE Reduction": "Baseline"
        },
        {
            "Configuration": "WITH NEWS (Topics & Sentiment)",
            "R2": res_topics["R2"],
            "Explained Variance": res_topics["Explained_Variance"],
            "RMSE (R$/MWh)": res_topics["RMSE"],
            "MAE (R$/MWh)": res_topics["MAE"],
            "MAPE (%)": res_topics["MAPE"],
            "R2 Lift": f"{(res_topics['R2'] - res_mkt['R2'])*100:+.2f}%",
            "RMSE Reduction": f"{(res_mkt['RMSE'] - res_topics['RMSE'])/res_mkt['RMSE']*100:+.2f}%"
        },
        {
            "Configuration": "WITH NEWS (Full: Topics + PCA)",
            "R2": res_full["R2"],
            "Explained Variance": res_full["Explained_Variance"],
            "RMSE (R$/MWh)": res_full["RMSE"],
            "MAE (R$/MWh)": res_full["MAE"],
            "MAPE (%)": res_full["MAPE"],
            "R2 Lift": f"{(res_full['R2'] - res_mkt['R2'])*100:+.2f}%",
            "RMSE Reduction": f"{(res_mkt['RMSE'] - res_full['RMSE'])/res_mkt['RMSE']*100:+.2f}%"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Top Feature Importances from Full Pipeline
    rf_full = res_full["Model"]
    feat_imp = pd.Series(rf_full.feature_importances_, index=X_train_full.columns).sort_values(ascending=False)
    print("\nTop 15 Most Important Features in Full Model:")
    for feat, val in feat_imp.head(15).items():
        print(f"   - {feat:<30}: {val:.4f} ({val*100:.2f}%)")

    # Save results to json
    out_results = {
        "split_method": "stratified_by_topic",
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "without_news": {k: v for k, v in res_mkt.items() if k != "Model"},
        "with_news_topics": {k: v for k, v in res_topics.items() if k != "Model"},
        "with_news_full": {k: v for k, v in res_full.items() if k != "Model"},
        "top_features": feat_imp.head(15).to_dict()
    }
    out_path = Path("results/rf_stratified_split_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_results, f, indent=4)
    print(f"\nSaved detailed results to {out_path}")

if __name__ == '__main__':
    main()
