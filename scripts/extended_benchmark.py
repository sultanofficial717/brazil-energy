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
REPORT_PATH = Path(get_result_path('extended_model_report.json'))


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return df


def metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def filter_weather_and_new(df):
    cols = [c for c in df.columns if not (c.startswith('weather_') or c.startswith('news_') or c.startswith('target_rolling_'))]
    return df[cols]


def aggregate_3d(df):
    d = df.copy().sort_values('date').reset_index(drop=True)
    d['_group'] = np.arange(len(d)) // 3
    feature_cols = [c for c in d.columns if c not in ['date', 'target_pld_next_day', '_group']]
    out = d.groupby('_group').agg({**{c: 'mean' for c in feature_cols}, 'target_pld_next_day': 'mean'}).reset_index(drop=True)
    return out


def build_features(df, remove_date=False, drop_weather_new=False, add_engineered=False):
    d = df.copy()
    target = 'target_pld_next_day'
    if 'date' in d.columns:
        if remove_date:
            d = d.drop(columns=['date'])
        else:
            d['date_num'] = d['date'].astype('int64') // 10**9
            d = d.drop(columns=['date'])
    if drop_weather_new:
        d = filter_weather_and_new(d)
    if add_engineered:
        d['target_rolling_mean_3d'] = d[target].shift(1).rolling(window=3, min_periods=1).mean()
        d['target_rolling_std_3d'] = d[target].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
    X = d.drop(columns=[target], errors='ignore')
    y = d[target]
    return X, y


def time_split(X, y, test_fraction=0.2):
    cutoff = int(len(X) * (1 - test_fraction))
    return X.iloc[:cutoff].copy(), X.iloc[cutoff:].copy(), y.iloc[:cutoff].copy(), y.iloc[cutoff:].copy()


def train_rf(X_train, X_test, y_train, y_test, with_pca=False):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if with_pca:
        steps.append(('scaler', StandardScaler()))
        steps.append(('pca', PCA(n_components=0.95, random_state=42)))
    steps.append(('rf', RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return metrics(y_test, pred)


def train_xgb(X_train, X_test, y_train, y_test, with_pca=False):
    prep = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    if with_pca:
        prep.steps.append(('scaler', StandardScaler()))
        prep.steps.append(('pca', PCA(n_components=0.95, random_state=42)))
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    reg.fit(Xtr, y_train)
    pred = reg.predict(Xte)
    return metrics(y_test, pred)


def train_lstm(X_train, X_test, y_train, y_test, with_pca=False):
    prep = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    if with_pca:
        prep.steps.append(('scaler', StandardScaler()))
        prep.steps.append(('pca', PCA(n_components=min(0.95, X_train.shape[1] - 1), random_state=42)))
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
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.LSTM(10),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xtr_seq, ytr_seq, epochs=3, batch_size=32, verbose=0)
    pred = model.predict(Xte_seq, verbose=0).ravel()
    return metrics(yte_seq, pred)


def run_experiments():
    df = load_data()
    configs = [
        ('daily_raw', dict(remove_date=False, drop_weather_new=False, agg=False, pca=False, engineered=False)),
        ('daily_raw_pca', dict(remove_date=False, drop_weather_new=False, agg=False, pca=True, engineered=False)),
        ('daily_no_date', dict(remove_date=True, drop_weather_new=False, agg=False, pca=False, engineered=False)),
        ('daily_no_weather_new', dict(remove_date=False, drop_weather_new=True, agg=False, pca=False, engineered=False)),
        ('daily_no_weather_new_pca', dict(remove_date=False, drop_weather_new=True, agg=False, pca=True, engineered=False)),
        ('daily_engineered', dict(remove_date=False, drop_weather_new=False, agg=False, pca=False, engineered=True)),
        ('three_day_agg', dict(remove_date=False, drop_weather_new=False, agg=True, pca=False, engineered=False)),
        ('three_day_agg_no_weather_new', dict(remove_date=False, drop_weather_new=True, agg=True, pca=False, engineered=False)),
    ]

    results = []
    for name, cfg in configs:
        work = df.copy()
        if cfg['agg']:
            work = aggregate_3d(work)
        X, y = build_features(work, remove_date=cfg['remove_date'], drop_weather_new=cfg['drop_weather_new'], add_engineered=cfg['engineered'])
        X_train, X_test, y_train, y_test = time_split(X, y)
        entry = {'config': name, 'metrics': {}}
        for model_name, fn in [('RandomForest', train_rf), ('XGBoost', train_xgb), ('LSTM', train_lstm)]:
            try:
                v = fn(X_train, X_test, y_train, y_test, with_pca=cfg['pca'])
                entry['metrics'][model_name] = v
                print(f'{name} | {model_name} | {v}')
            except Exception as e:
                entry['metrics'][model_name] = {'error': str(e)}
                print(f'{name} | {model_name} | ERROR: {e}')
        results.append(entry)

    best = None
    for item in results:
        for model, vals in item['metrics'].items():
            if 'error' in vals:
                continue
            score = (vals['rmse'], vals['mae'])
            if best is None or score < best[0]:
                best = (score, item['config'], model, vals)
    report = {'best_model': {'config': best[1], 'model': best[2], 'metrics': best[3]}, 'all_results': results}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nBEST MODEL:', report['best_model'])
    print(f'REPORT SAVED: {REPORT_PATH}')
    return report


if __name__ == '__main__':
    run_experiments()
