import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))
from paths import get_data_path, get_figure_path, get_result_path, get_report_path

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

warnings.filterwarnings('ignore')

DATA_PATH = Path(get_data_path('daily_train_ready.csv'))
REPORT_PATH = Path(get_result_path('model_comparison_report.json'))


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return df


def time_split(X, y, test_fraction=0.2):
    cutoff = int(len(X) * (1 - test_fraction))
    return X.iloc[:cutoff].copy(), X.iloc[cutoff:].copy(), y.iloc[:cutoff].copy(), y.iloc[cutoff:].copy()


def metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def aggregate_3d(df):
    d = df.copy().sort_values('date').reset_index(drop=True)
    d['_group'] = np.arange(len(d)) // 3
    feature_cols = [c for c in d.columns if c not in ['date', 'target_pld_next_day', '_group']]
    out = d.groupby('_group').agg({**{c: 'mean' for c in feature_cols}, 'target_pld_next_day': 'mean'}).reset_index(drop=True)
    return out


def build_dataset(df, remove_date=False, add_feature_engineering=False):
    d = df.copy()
    target = 'target_pld_next_day'

    if 'date' in d.columns:
        if remove_date:
            d = d.drop(columns=['date'])
        else:
            d['date_num'] = d['date'].astype('int64') // 10**9
            d = d.drop(columns=['date'])

    if add_feature_engineering:
        d['target_rolling_mean_3d'] = d[target].shift(1).rolling(window=3, min_periods=1).mean()
        d['target_rolling_std_3d'] = d[target].shift(1).rolling(window=3, min_periods=1).std().fillna(0)

    X = d.drop(columns=[target])
    y = d[target]
    return X, y


def train_random_forest(X_train, X_test, y_train, y_test, with_pca=False):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if with_pca:
        steps.append(('scaler', StandardScaler()))
        steps.append(('pca', PCA(n_components=0.95, random_state=42)))
    steps.append(('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return metrics(y_test, pred)


def train_xgboost(X_train, X_test, y_train, y_test, with_pca=False):
    prep = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    if with_pca:
        prep.steps.append(('scaler', StandardScaler()))
        prep.steps.append(('pca', PCA(n_components=0.95, random_state=42)))
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_test)
    reg = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(Xt, y_train)
    pred = reg.predict(Xv)
    return metrics(y_test, pred)


def train_lstm(X_train, X_test, y_train, y_test, with_pca=False):
    prep = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    if with_pca:
        prep.steps.append(('scaler', StandardScaler()))
        prep.steps.append(('pca', PCA(n_components=min(0.95, X_train.shape[1]-1), random_state=42)))
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)

    seq_len = 3
    def make_seq(arr, target):
        seqs, labs = [], []
        for i in range(len(arr) - seq_len + 1):
            seqs.append(arr[i:i + seq_len])
            labs.append(target.iloc[i + seq_len - 1])
        return np.asarray(seqs, dtype=np.float32), np.asarray(labs, dtype=np.float32)

    Xtr_seq, ytr_seq = make_seq(Xtr, y_train)
    Xte_seq, yte_seq = make_seq(Xte, y_test)
    if Xtr_seq.shape[0] == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, Xtr_seq.shape[2])),
        keras.layers.LSTM(16, return_sequences=True),
        keras.layers.LSTM(8),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xtr_seq, ytr_seq, epochs=4, batch_size=32, verbose=0)
    pred = model.predict(Xte_seq, verbose=0).ravel()
    return metrics(yte_seq, pred)


def run_benchmark(df):
    configs = [
        ('raw_daily', dict(remove_date=False, engineered=False, agg=False, pca=False)),
        ('raw_daily_pca', dict(remove_date=False, engineered=False, agg=False, pca=True)),
        ('date_removed', dict(remove_date=True, engineered=False, agg=False, pca=False)),
        ('date_removed_pca', dict(remove_date=True, engineered=False, agg=False, pca=True)),
        ('feature_engineered', dict(remove_date=False, engineered=True, agg=False, pca=False)),
        ('feature_engineered_pca', dict(remove_date=False, engineered=True, agg=False, pca=True)),
        ('aggregated_3d', dict(remove_date=False, engineered=False, agg=True, pca=False)),
        ('aggregated_3d_pca', dict(remove_date=False, engineered=False, agg=True, pca=True)),
    ]

    results = []
    for name, cfg in configs:
        work = df.copy()
        if cfg['agg']:
            work = aggregate_3d(work)
        X, y = build_dataset(work, remove_date=cfg['remove_date'], add_feature_engineering=cfg['engineered'])
        X_train, X_test, y_train, y_test = time_split(X, y)

        config_result = {'config': name, 'metrics': {}}
        for model_name, fn in [('RandomForest', train_random_forest), ('XGBoost', train_xgboost), ('LSTM', train_lstm)]:
            try:
                m = fn(X_train, X_test, y_train, y_test, with_pca=cfg['pca'])
                config_result['metrics'][model_name] = m
                print(f'{name} | {model_name} | {m}')
            except Exception as e:
                config_result['metrics'][model_name] = {'error': str(e)}
                print(f'{name} | {model_name} | ERROR: {e}')
        results.append(config_result)

    best = None
    for item in results:
        for model, vals in item['metrics'].items():
            if 'error' in vals:
                continue
            candidate = (vals['rmse'], vals['mae'], item['config'], model)
            if best is None or candidate[:3] < best[0][:3]:
                best = (candidate, item['config'], model, vals)
    print('\nBEST_OVERALL', best[1], best[2], best[3])
    return results, best


def main():
    df = load_data()
    results, best = run_benchmark(df)
    report = {
        'best_model': {'config': best[1], 'model': best[2], 'metrics': best[3]},
        'all_results': results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nSaved report to {REPORT_PATH}')


if __name__ == '__main__':
    main()
