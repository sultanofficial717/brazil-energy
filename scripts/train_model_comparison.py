import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))
from paths import get_data_path, get_figure_path, get_result_path, get_report_path

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.base import clone
except Exception as e:
    raise SystemExit(f'Scikit-learn import failed: {e}')

try:
    import xgboost as xgb
except Exception as e:
    print(f'XGBoost import failed: {e}')
    xgb = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch device: {DEVICE}')
except Exception as e:
    print(f'PyTorch import failed: {e}')
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    DEVICE = None


DATA_PATH = Path(get_data_path('daily_train_ready.csv'))
OUTPUT_PATH = Path(get_result_path('model_comparison_report.json'))


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = df.copy()
    print('shape', df.shape)
    print('date range', df['date'].min(), df['date'].max())
    print('nulls', int(df.isna().sum().sum()))
    return df


def build_features(df, remove_date=False, add_rolling_target=False, include_news=True, include_weather=True):
    feature_df = df.copy()
    target = 'target_pld_next_day'
    if 'date' in feature_df.columns and not remove_date:
        feature_df['date_num'] = feature_df['date'].astype('int64') // 10**9
        feature_df = feature_df.drop(columns=['date'])
    elif remove_date:
        feature_df = feature_df.drop(columns=['date'], errors='ignore')

    if add_rolling_target:
        feature_df['target_rolling_mean_3d'] = feature_df[target].shift(1).rolling(window=3, min_periods=1).mean()
        feature_df['target_rolling_std_3d'] = feature_df[target].shift(1).rolling(window=3, min_periods=1).std().fillna(0)

    X = feature_df.drop(columns=[target], errors='ignore')
    
    # Filter features based on news/weather flags
    if not include_news:
        X = X.drop(columns=[c for c in X.columns if c.startswith('news_')], errors='ignore')
    if not include_weather:
        X = X.drop(columns=[c for c in X.columns if c.startswith('weather_')], errors='ignore')
    
    y = feature_df[target]
    return X, y


def make_time_split(X, y, test_fraction=0.2):
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def metric_dict(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }


def train_rf(X_train, X_test, y_train, y_test, pca=False):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    if pca:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('rf', model)
        ])
    else:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('rf', model)
        ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return metric_dict(y_test, pred), model


def train_xgb(X_train, X_test, y_train, y_test, pca=False):
    if xgb is None:
        return None, None
    
    # Use GPU if available
    tree_method = 'gpu_hist' if torch else 'hist'
    gpu_id = 0 if torch else None
    
    reg = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method=tree_method,
        gpu_id=gpu_id if gpu_id is not None else -1
    )
    if pca:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('xgb', reg)
        ])
    else:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('xgb', reg)
        ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return metric_dict(y_test, pred), model


def train_lstm(X_train, X_test, y_train, y_test, pca=False):
    if torch is None or nn is None or DEVICE is None:
        return None, None

    X_tr = X_train.to_numpy(dtype=np.float32)
    X_te = X_test.to_numpy(dtype=np.float32)
    y_tr = y_train.to_numpy(dtype=np.float32)
    y_te = y_test.to_numpy(dtype=np.float32)

    prep = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_tr = prep.fit_transform(X_tr)
    X_te = prep.transform(X_te)

    if pca:
        pca_model = PCA(n_components=min(0.95, X_tr.shape[1] - 1), random_state=42)
        X_tr = pca_model.fit_transform(X_tr)
        X_te = pca_model.transform(X_te)

    seq_len = 3
    if X_tr.shape[0] < seq_len:
        return None, None

    def build_seq(arr, target):
        seqs = []
        labels = []
        for i in range(len(arr) - seq_len + 1):
            seqs.append(arr[i:i + seq_len])
            labels.append(target[i + seq_len - 1])
        return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32)

    X_tr_seq, y_tr_seq = build_seq(X_tr, y_tr)
    X_te_seq, y_te_seq = build_seq(X_te, y_te)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 32, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model = LSTMRegressor(X_tr_seq.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq).view(-1, 1)), batch_size=32, shuffle=True)
    model.train()
    for epoch in range(12):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_te_seq_tensor = torch.tensor(X_te_seq).to(DEVICE)
        pred = model(X_te_seq_tensor).view(-1).cpu().numpy()

    return metric_dict(y_te_seq, pred), model


def aggregate_3d(df):
    agg = df.copy()
    agg = agg.sort_values('date').reset_index(drop=True)
    feature_cols = [c for c in agg.columns if c not in ['date', 'target_pld_next_day']]
    agg = agg.assign(_window_id=(np.arange(len(agg)) // 3)).copy()
    grouped = agg.groupby('_window_id')
    out = grouped[feature_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
    out.columns = ['_window_id'] + [f'{c}_{stat}' for c, stat in [
        (col, stat) for col in feature_cols for stat in ['mean', 'std', 'min', 'max']
    ]]
    target = grouped['target_pld_next_day'].mean().reset_index(name='target_pld_next_day')
    result = out.merge(target, on='_window_id', how='left')
    result = result.drop(columns=['_window_id'])
    return result


def run_experiments(df):
    results = []
    configs = [
        # Original raw daily configurations
        ('raw_daily_with_news_weather', dict(remove_date=False, add_rolling_target=False, pca=False, include_news=True, include_weather=True)),
        ('raw_daily_without_news_weather', dict(remove_date=False, add_rolling_target=False, pca=False, include_news=False, include_weather=False)),
        ('raw_daily_with_news_weather_pca', dict(remove_date=False, add_rolling_target=False, pca=True, include_news=True, include_weather=True)),
        ('raw_daily_without_news_weather_pca', dict(remove_date=False, add_rolling_target=False, pca=True, include_news=False, include_weather=False)),
        
        # Feature engineered with rolling targets
        ('feature_engineered_with_news_weather', dict(remove_date=False, add_rolling_target=True, pca=False, include_news=True, include_weather=True)),
        ('feature_engineered_without_news_weather', dict(remove_date=False, add_rolling_target=True, pca=False, include_news=False, include_weather=False)),
        ('feature_engineered_with_news_weather_pca', dict(remove_date=False, add_rolling_target=True, pca=True, include_news=True, include_weather=True)),
        ('feature_engineered_without_news_weather_pca', dict(remove_date=False, add_rolling_target=True, pca=True, include_news=False, include_weather=False)),
    ]

    for label, settings in configs:
        print(f'\n=== {label} ===')
        data = df.copy()
        X, y = build_features(
            data, 
            remove_date=settings['remove_date'], 
            add_rolling_target=settings['add_rolling_target'],
            include_news=settings.get('include_news', True),
            include_weather=settings.get('include_weather', True)
        )

        X_train, X_test, y_train, y_test = make_time_split(X, y, test_fraction=0.2)
        print(f'Features: {X.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')
        
        metrics = {}
        for model_name, trainer in [('RandomForest', train_rf), ('XGBoost', train_xgb), ('LSTM', train_lstm)]:
            try:
                m, _ = trainer(X_train, X_test, y_train, y_test, pca=settings['pca'])
                if m is not None:
                    metrics[model_name] = m
                    print(f'{model_name}: {m}')
            except Exception as e:
                print(f'{model_name} failed: {e}')
                metrics[model_name] = {'error': str(e)}
        results.append({
            'config': label,
            'remove_date': settings['remove_date'],
            'pca': settings['pca'],
            'include_news': settings.get('include_news', True),
            'include_weather': settings.get('include_weather', True),
            'engineered_features': settings['add_rolling_target'],
            'num_features': X.shape[1],
            'metrics': metrics,
        })

    return results


def save_report(results):
    best = None
    best_score = None
    for item in results:
        for model, metrics in item['metrics'].items():
            if 'error' in metrics:
                continue
            score = metrics['rmse']
            key = (score, metrics['mae'])
            if best is None or key < best_score:
                best = {'config': item['config'], 'model': model, 'metrics': metrics}
                best_score = key
    report = {
        'best_model': best,
        'all_results': results,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('\nBEST MODEL:', best)
    print('Saved report to', OUTPUT_PATH)
    return report


if __name__ == '__main__':
    df = load_data()
    results = run_experiments(df)
    save_report(results)
