import logging
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, r2_score

logger = logging.getLogger(__name__)

WRIS_BASE = os.environ.get('WRIS_BASE_URL', 'https://indiawris.gov.in')
WRIS_API_BASE = os.environ.get('WRIS_API_BASE', 'https://indiawris.gov.in/wris')

DEFAULT_AGENCY = 'CGWB'
DEFAULT_SIZE = 500
DEFAULT_DATE_RANGE_DAYS = 365

STATE_LIST_PATH = '/getStatesListDataViewPage/'
DISTRICT_LIST_PATH = '/getAllDistrictListDataViewPage/{state}/'
AGENCY_LIST_PATH = '/getMasterAgencyListDataViewPage/'
DATA_PATH = '/Dataset/Ground%20Water%20Level'


def safe_get(url, **kwargs):
    try:
        resp = requests.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning('GET failed %s: %s', url, exc)
        return None


def safe_post(url, data=None, json=None, headers=None):
    try:
        resp = requests.post(url, data=data, json=json, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning('POST failed %s: %s', url, exc)
        return None


def fetch_states():
    url = WRIS_API_BASE + STATE_LIST_PATH
    resp = safe_get(url)
    if resp is None:
        return []
    try:
        data = resp.json()
        if isinstance(data, list):
            return [str(item.get('state') or item.get('State') or item.get('name') or item.get('stateName')).strip() for item in data if item]
        if isinstance(data, dict) and 'data' in data:
            return [str(item).strip() for item in data['data'] if item]
    except Exception as exc:
        logger.warning('Could not decode state list JSON: %s', exc)
    return []


def fetch_districts(state):
    if not state:
        return []
    url = WRIS_API_BASE + DISTRICT_LIST_PATH.format(state=requests.utils.requote_uri(state))
    resp = safe_get(url)
    if resp is None:
        return []
    try:
        data = resp.json()
        if isinstance(data, list):
            return [str(item.get('district') or item.get('District') or item.get('name') or item.get('districtName')).strip() for item in data if item]
        if isinstance(data, dict) and 'data' in data:
            return [str(item).strip() for item in data['data'] if item]
    except Exception as exc:
        logger.warning('Could not decode district list JSON for %s: %s', state, exc)
    return []


def fetch_agencies():
    url = WRIS_API_BASE + AGENCY_LIST_PATH
    resp = safe_get(url)
    if resp is None:
        return []
    try:
        data = resp.json()
        if isinstance(data, list):
            return [str(item.get('agency') or item.get('Agency') or item.get('name') or item.get('agencyName')).strip() for item in data if item]
        if isinstance(data, dict) and 'data' in data:
            return [str(item).strip() for item in data['data'] if item]
    except Exception as exc:
        logger.warning('Could not decode agency list JSON: %s', exc)
    return []


def normalize_groundwater_frame(df):
    if df is None or df.empty:
        return None

    time_col = _safe_column(df, ['dataTime', 'date', 'timestamp', 'recordDate'])
    value_col = _safe_column(df, ['dataValue', 'value', 'waterLevel', 'groundwater_level'])

    if time_col is None or value_col is None:
        return None

    frame = df[[time_col, value_col]].copy()
    frame.columns = ['date_raw', 'level_raw']
    frame['date'] = pd.to_datetime(frame['date_raw'], errors='coerce')
    frame['groundwater_level'] = pd.to_numeric(frame['level_raw'], errors='coerce')
    frame = frame.dropna(subset=['date', 'groundwater_level'])
    if frame.empty:
        return None
    frame = frame.sort_values('date')
    frame = frame.groupby('date', as_index=False)['groundwater_level'].mean()
    frame = frame.set_index('date')
    frame = frame[~frame.index.duplicated(keep='last')]
    return frame


def _safe_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def build_features(frame):
    data = frame.copy()
    for lag in [1, 3, 7, 14]:
        data[f'lag_{lag}'] = data['groundwater_level'].shift(lag)
    data['rolling_7'] = data['groundwater_level'].rolling(7, min_periods=1).mean()
    data['rolling_14'] = data['groundwater_level'].rolling(14, min_periods=1).mean()
    data['rolling_7_std'] = data['groundwater_level'].rolling(7, min_periods=1).std().fillna(0)
    data['rolling_14_std'] = data['groundwater_level'].rolling(14, min_periods=1).std().fillna(0)
    data['diff_1'] = data['groundwater_level'].diff(1)
    data['diff_7'] = data['groundwater_level'].diff(7)
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear
    data['weekday'] = data.index.weekday
    data['is_monsoon'] = data.index.month.isin([6, 7, 8, 9]).astype(int)
    feature_cols = [
        'lag_1', 'lag_3', 'lag_7', 'lag_14', 'rolling_7', 'rolling_14',
        'rolling_7_std', 'rolling_14_std', 'diff_1', 'diff_7',
        'month', 'dayofyear', 'weekday', 'is_monsoon',
    ]
    data = data.dropna(subset=['lag_14', 'diff_7'])
    return data, feature_cols


def evaluate_research_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    spread = float(np.ptp(y_true)) or 1.0
    mean_true = float(np.mean(y_true)) or 1.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / np.maximum(np.abs(y_true), 1e-6))) * 100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    nse = float(1.0 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) > 1 and not np.isclose(np.sum((y_true - np.mean(y_true)) ** 2), 0.0) else np.nan
    nrmse = float(rmse / spread)
    pbias = float(100.0 * np.sum(y_pred - y_true) / np.maximum(np.sum(y_true), 1e-6))
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and not np.allclose(np.std(y_true), 0.0) and not np.allclose(np.std(y_pred), 0.0) else np.nan
    bias = float(np.mean(y_pred - y_true))
    if len(y_true) > 1 and not np.allclose(np.std(y_true), 0.0) and not np.allclose(np.std(y_pred), 0.0):
        alpha = float(np.std(y_pred) / np.std(y_true))
        beta = float((np.mean(y_pred) or 1.0) / mean_true)
        kge = float(1.0 - np.sqrt((pearson_r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        alpha = np.nan
        beta = np.nan
        kge = np.nan
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'NSE': nse,
        'NRMSE': nrmse,
        'PBIAS': pbias,
        'Pearson r': pearson_r,
        'Bias': bias,
        'KGE': kge,
        'Alpha': alpha,
        'Beta': beta,
    }


def build_regime_bins(train_series):
    q1 = float(np.quantile(train_series, 0.33))
    q2 = float(np.quantile(train_series, 0.66))
    if np.isclose(q1, q2):
        spread = float(np.std(train_series)) or 1.0
        q1 = float(np.median(train_series) - spread * 0.25)
        q2 = float(np.median(train_series) + spread * 0.25)
    if q1 >= q2:
        q2 = q1 + (abs(q1) * 0.05 + 1e-6)
    return [-np.inf, q1, q2, np.inf], ['Low', 'Moderate', 'High']


def make_regime_labels(values, bins, labels):
    return pd.cut(pd.Series(values), bins=bins, labels=labels, include_lowest=True)


def regime_confusion_matrix(y_true, y_pred, bins, labels):
    true_classes = make_regime_labels(y_true, bins, labels)
    pred_classes = make_regime_labels(y_pred, bins, labels)
    matrix = confusion_matrix(true_classes, pred_classes, labels=labels)
    return pd.DataFrame(matrix, index=[f'Actual {label}' for label in labels], columns=[f'Pred {label}' for label in labels])


def fetch_groundwater_data(state, district, agency=DEFAULT_AGENCY, startdate=None, enddate=None, size=DEFAULT_SIZE):
    if startdate is None:
        enddate = datetime.now().date()
        startdate = enddate - timedelta(days=DEFAULT_DATE_RANGE_DAYS)
    if enddate is None:
        enddate = datetime.now().date()
    if isinstance(startdate, datetime):
        startdate = startdate.strftime('%Y-%m-%d')
    if isinstance(enddate, datetime):
        enddate = enddate.strftime('%Y-%m-%d')
    payload = {
        'stateName': state,
        'districtName': district,
        'agencyName': agency,
        'startdate': startdate,
        'enddate': enddate,
        'download': 'false',
        'page': 0,
        'size': size,
    }
    url = WRIS_BASE + DATA_PATH
    resp = safe_post(url, data=payload, headers={'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'})
    if resp is None:
        return None
    try:
        payload = resp.json()
    except Exception as exc:
        logger.warning('Failed to parse JSON from groundwater data: %s', exc)
        return None
    data = payload.get('data') or payload.get('Data') or []
    if not data:
        return None
    return pd.DataFrame(data)


def time_series_split(frame, train_ratio=0.8):
    split_index = max(1, int(len(frame) * train_ratio))
    return frame.iloc[:split_index], frame.iloc[split_index:]


def find_best_split_ratio(featured, feature_cols):
    for ratio in [0.80, 0.75, 0.70]:
        train_df, test_df = time_series_split(featured, train_ratio=ratio)
        if test_df.empty or len(test_df) < 2 or len(train_df) < 5:
            continue
        regime_bins, regime_labels = build_regime_bins(train_df['groundwater_level'].values)
        actual_regime_labels = make_regime_labels(test_df['groundwater_level'].values, regime_bins, regime_labels)
        if actual_regime_labels.nunique(dropna=True) == 3:
            return ratio, regime_bins, regime_labels, 3
    return 0.80, None, None, 0


def train_models(featured, feature_cols, split_ratio=0.80):
    if len(featured) < 20:
        return None
    train_df, test_df = time_series_split(featured, train_ratio=split_ratio)
    if test_df.empty:
        return None
    X_train = train_df[feature_cols]
    y_train = train_df['groundwater_level']
    X_test = test_df[feature_cols]
    y_test = test_df['groundwater_level']
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    xgb = XGBRegressor(n_estimators=350, learning_rate=0.04, max_depth=5, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, random_state=42, objective='reg:squarederror', n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    persistence_pred = np.repeat(y_train.iloc[-1], len(y_test))
    metrics = pd.DataFrame({
        'Persistence': evaluate_research_metrics(y_test.values, persistence_pred),
        'Random Forest': evaluate_research_metrics(y_test.values, rf_pred),
        'XGBoost': evaluate_research_metrics(y_test.values, xgb_pred),
    }).T
    regime_bins, regime_labels = build_regime_bins(y_train.values)
    actual_regime_labels = make_regime_labels(y_test.values, regime_bins, regime_labels)
    regime_class_count = int((actual_regime_labels.value_counts(dropna=False) > 0).sum())
    persistence_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(persistence_pred, regime_bins, regime_labels)).mean())
    rf_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(rf_pred, regime_bins, regime_labels)).mean())
    xgb_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(xgb_pred, regime_bins, regime_labels)).mean())
    persistence_cm = regime_confusion_matrix(y_test.values, persistence_pred, regime_bins, regime_labels)
    rf_cm = regime_confusion_matrix(y_test.values, rf_pred, regime_bins, regime_labels)
    xgb_cm = regime_confusion_matrix(y_test.values, xgb_pred, regime_bins, regime_labels)
    anomalies = IsolationForest(contamination=0.05, random_state=42)
    anomaly_input = featured[['groundwater_level', 'rolling_7', 'rolling_14', 'diff_1', 'diff_7']].copy()
    featured_copy = featured.copy()
    featured_copy['anomaly_flag'] = anomalies.fit_predict(anomaly_input)
    featured_copy['anomaly_flag'] = (featured_copy['anomaly_flag'] == -1).astype(int)
    featured_copy['anomaly_score'] = anomalies.decision_function(anomaly_input)
    metrics['Regime Acc'] = [persistence_regime_accuracy, rf_regime_accuracy, xgb_regime_accuracy]
    metrics['Regime Acc'] = metrics['Regime Acc'].astype(float)
    metrics['Regime Coverage'] = float(len(actual_regime_labels.dropna().unique())) / 3.0
    metrics['Regime Valid'] = float(regime_class_count == 3)
    return {
        'train_df': train_df,
        'test_df': test_df,
        'rf': rf,
        'xgb': xgb,
        'rf_pred': rf_pred,
        'xgb_pred': xgb_pred,
        'persistence_pred': persistence_pred,
        'metrics': metrics,
        'featured': featured_copy,
        'feature_cols': feature_cols,
        'interval_low': float(np.quantile(y_test.values - xgb_pred, 0.1)),
        'interval_high': float(np.quantile(y_test.values - xgb_pred, 0.9)),
        'regime_bins': regime_bins,
        'regime_labels': regime_labels,
        'regime_confusion_matrices': {
            'Persistence': persistence_cm,
            'Random Forest': rf_cm,
            'XGBoost': xgb_cm,
        },
        'regime_valid': regime_class_count == 3,
        'regime_class_count': regime_class_count,
    }


def build_district_metrics(state, district, agency=DEFAULT_AGENCY, startdate=None, enddate=None, size=DEFAULT_SIZE):
    raw_df = fetch_groundwater_data(state, district, agency, startdate, enddate, size)
    if raw_df is None or raw_df.empty:
        return None
    featured = normalize_groundwater_frame(raw_df)
    if featured is None or featured.empty:
        return None
    featured, feature_cols = build_features(featured)
    if featured is None or featured.empty:
        return None
    split_ratio, regime_bins, regime_labels, regime_class_count = find_best_split_ratio(featured, feature_cols)
    trained = train_models(featured, feature_cols, split_ratio=split_ratio)
    if trained is None:
        return None
    return {
        'state': state,
        'district': district,
        'agency': agency,
        'rows': len(featured),
        'train_rows': len(trained['train_df']),
        'test_rows': len(trained['test_df']),
        'split_ratio': split_ratio,
        'regime_valid': trained['regime_valid'],
        'regime_class_count': trained['regime_class_count'],
        'anomaly_rate_pct': float(trained['featured']['anomaly_flag'].mean() * 100),
        'metrics': trained['metrics'],
        'raw_df': raw_df,
        'featured': trained['featured'],
        'regime_confusion_matrices': trained['regime_confusion_matrices'],
    }


def aggregate_metrics(df, weight_col='rows'):
    if df is None or df.empty:
        return None
    metrics = []
    for _, row in df.iterrows():
        for model_name in ['Persistence', 'Random Forest', 'XGBoost']:
            prefix = f'{model_name}__'
            metric_values = {k[len(prefix):]: row[k] for k in df.columns if k.startswith(prefix)}
            metric_values['state'] = row['state']
            metric_values['district'] = row['district']
            metric_values['model'] = model_name
            metric_values['weight'] = row.get(weight_col, 1)
            metrics.append(metric_values)
    result = pd.DataFrame(metrics)
    results = []
    for model_name, group in result.groupby('model'):
        summary = {'model': model_name}
        weight = group['weight'].fillna(1.0)
        for metric in ['RMSE', 'MAPE', 'R2', 'NSE', 'NRMSE', 'PBIAS', 'KGE', 'Regime Acc']:
            if metric not in group.columns:
                continue
            if metric in ['RMSE', 'MAE', 'MAPE', 'PBIAS', 'Bias']:
                summary[metric] = float((group[metric].fillna(0.0) * weight).sum() / max(weight.sum(), 1.0))
            elif metric in ['R2', 'NSE', 'Pearson r', 'KGE', 'Regime Acc']:
                summary[metric] = float((group[metric].fillna(0.0) * weight).sum() / max(weight.sum(), 1.0))
            else:
                summary[metric] = float(group[metric].median(skipna=True))
        summary['num_districts'] = int(group['district'].nunique())
        results.append(summary)
    return pd.DataFrame(results)
