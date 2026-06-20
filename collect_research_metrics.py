import argparse
from datetime import datetime, timedelta
import os
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Keep this script separate from the Flask app and Streamlit dashboard.
# It hits the backend endpoints for states and districts, fetches groundwater data,
# computes the same research metrics, and writes them to a CSV.

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
DEFAULT_OUTPUT = os.environ.get("RESEARCH_SWEEP_CSV", "research_metrics_sweep.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect research metrics for all states and districts.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="URL of the Flask backend (example: http://localhost:5000)")
    parser.add_argument("--agency", default=None, help="Agency to use for all district queries. If omitted, uses CGWB or the first agency returned.")
    parser.add_argument("--start-date", default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"), help="Start date for groundwater data retrieval.")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), help="End date for groundwater data retrieval.")
    parser.add_argument("--size", type=int, default=500, help="Max records to request per district.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV file to write the collected metrics.")
    parser.add_argument("--append-output", default=os.environ.get("RESEARCH_APPEND_CSV", "research_metrics_append.csv"), help="CSV file to append collected metrics for longer-term accumulation.")
    parser.add_argument("--state", default=None, help="Optional single state to run instead of all states.")
    parser.add_argument("--district", default=None, help="Optional single district to run instead of all districts in a state.")
    return parser.parse_args()


def request_json(session, url, params=None):
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_states(session, backend_url):
    url = f"{backend_url.rstrip('/')}/states"
    payload = request_json(session, url)
    return payload.get("states", []) if payload.get("status") == "success" else []


def fetch_districts(session, backend_url, state):
    url = f"{backend_url.rstrip('/')}/districts/{requests.utils.requote_uri(state)}"
    payload = request_json(session, url)
    return payload.get("districts", []) if payload.get("status") == "success" else []


def fetch_agencies(session, backend_url):
    url = f"{backend_url.rstrip('/')}/agencies"
    payload = request_json(session, url)
    return payload.get("agencies", []) if payload.get("status") == "success" else []


def fetch_groundwater_data(session, backend_url, state, district, agency, startdate, enddate, size=500):
    url = f"{backend_url.rstrip('/')}/groundwater"
    params = {
        "state": state,
        "district": district,
        "agency": agency,
        "startdate": startdate,
        "enddate": enddate,
        "size": size,
    }
    payload = request_json(session, url, params=params)
    if payload.get("status") != "success":
        raise ValueError(payload.get("message", "Unexpected backend response"))
    data = payload.get("data", [])
    return pd.DataFrame(data)


def _safe_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def normalize_groundwater_frame(df):
    if df is None or df.empty:
        return None

    time_col = _safe_column(df, ["dataTime", "date", "timestamp", "recordDate"])
    value_col = _safe_column(df, ["dataValue", "value", "waterLevel", "groundwater_level"])

    if time_col is None or value_col is None:
        return None

    frame = df[[time_col, value_col]].copy()
    frame.columns = ["date_raw", "level_raw"]
    frame["date"] = pd.to_datetime(frame["date_raw"], errors="coerce")
    frame["groundwater_level"] = pd.to_numeric(frame["level_raw"], errors="coerce")
    frame = frame.dropna(subset=["date", "groundwater_level"])
    frame = frame.sort_values("date")
    frame = frame.groupby("date", as_index=False)["groundwater_level"].mean()
    frame = frame.set_index("date")
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


FEATURE_COLS = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_7",
    "rolling_14",
    "rolling_7_std",
    "rolling_14_std",
    "diff_1",
    "diff_7",
    "month",
    "dayofyear",
    "weekday",
    "is_monsoon",
]


def build_features(frame):
    data = frame.copy()
    for lag in [1, 3, 7, 14]:
        data[f"lag_{lag}"] = data["groundwater_level"].shift(lag)
    data["rolling_7"] = data["groundwater_level"].rolling(7, min_periods=1).mean()
    data["rolling_14"] = data["groundwater_level"].rolling(14, min_periods=1).mean()
    data["rolling_7_std"] = data["groundwater_level"].rolling(7, min_periods=1).std().fillna(0)
    data["rolling_14_std"] = data["groundwater_level"].rolling(14, min_periods=1).std().fillna(0)
    data["diff_1"] = data["groundwater_level"].diff(1)
    data["diff_7"] = data["groundwater_level"].diff(7)
    data["month"] = data.index.month
    data["dayofyear"] = data.index.dayofyear
    data["weekday"] = data.index.weekday
    data["is_monsoon"] = data.index.month.isin([6, 7, 8, 9]).astype(int)

    data = data.dropna(subset=["lag_14", "diff_7"])
    return data, FEATURE_COLS


def time_series_split(frame, train_ratio=0.8):
    split_index = max(1, int(len(frame) * train_ratio))
    return frame.iloc[:split_index], frame.iloc[split_index:]


def evaluate_research_metrics(y_true, y_pred, model_name=None, district=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    gwl_min = float(np.min(y_true)) if len(y_true) else np.nan
    gwl_max = float(np.max(y_true)) if len(y_true) else np.nan
    gwl_range = float(gwl_max - gwl_min)
    mean_true = float(np.mean(y_true)) if len(y_true) else np.nan

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / np.maximum(np.abs(y_true), 1e-6))) * 100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    nse = float(1.0 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) > 1 and not np.isclose(np.sum((y_true - np.mean(y_true)) ** 2), 0.0) else np.nan
    if gwl_range > 0:
        nrmse = float(rmse / gwl_range)
        if nrmse > 1.0:
            warning_name = f" for district '{district}'" if district else ""
            warning_model = f" ({model_name})" if model_name else ""
            print(f"      WARNING: computed NRMSE{warning_model}{warning_name} = {nrmse:.4f} > 1.0")
    else:
        nrmse = np.nan

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
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "NSE": nse,
        "NRMSE": nrmse,
        "PBIAS": pbias,
        "Pearson r": pearson_r,
        "Bias": bias,
        "KGE": kge,
        "Alpha": alpha,
        "Beta": beta,
        "gwl_min": gwl_min,
        "gwl_max": gwl_max,
        "gwl_range": gwl_range,
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
    return [-np.inf, q1, q2, np.inf], ["Low", "Moderate", "High"]


def make_regime_labels(values, bins, labels):
    return pd.cut(pd.Series(values), bins=bins, labels=labels, include_lowest=True)


def regime_confusion_matrix(y_true, y_pred, bins, labels):
    true_classes = make_regime_labels(y_true, bins, labels)
    pred_classes = make_regime_labels(y_pred, bins, labels)
    matrix = confusion_matrix(true_classes, pred_classes, labels=labels)
    return pd.DataFrame(matrix, index=[f"Actual {label}" for label in labels], columns=[f"Pred {label}" for label in labels])


def train_models(featured, feature_cols, district=None):
    if len(featured) < 20:
        return None

    train_df, test_df = time_series_split(featured)
    if test_df.empty:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df["groundwater_level"]
    X_test = test_df[feature_cols]
    y_test = test_df["groundwater_level"]
    X_full = featured[feature_cols]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    persistence_pred = np.repeat(y_train.iloc[-1], len(y_test))

    regime_bins, regime_labels = build_regime_bins(y_train.values)
    actual_regime = make_regime_labels(y_test.values, regime_bins, regime_labels)
    persistence_regime = make_regime_labels(persistence_pred, regime_bins, regime_labels)
    rf_regime = make_regime_labels(rf_pred, regime_bins, regime_labels)
    xgb_regime = make_regime_labels(xgb_pred, regime_bins, regime_labels)

    actual_regime_counts = actual_regime.value_counts(dropna=False).reindex(regime_labels, fill_value=0)
    regime_class_count = int((actual_regime_counts > 0).sum())

    persistence_regime_accuracy = float((actual_regime == persistence_regime).mean())
    rf_regime_accuracy = float((actual_regime == rf_regime).mean())
    xgb_regime_accuracy = float((actual_regime == xgb_regime).mean())

    persistence_cm = regime_confusion_matrix(y_test.values, persistence_pred, regime_bins, regime_labels)
    rf_cm = regime_confusion_matrix(y_test.values, rf_pred, regime_bins, regime_labels)
    xgb_cm = regime_confusion_matrix(y_test.values, xgb_pred, regime_bins, regime_labels)

    residuals = y_test.values - xgb_pred
    interval_low = float(np.quantile(residuals, 0.1))
    interval_high = float(np.quantile(residuals, 0.9))

    anomalies = IsolationForest(contamination=0.05, random_state=42)
    anomalies.fit(X_train)
    anomaly_labels = anomalies.predict(X_full)
    anomaly_scores = anomalies.decision_function(X_full)

    featured = featured.copy()
    featured["anomaly_flag"] = (anomaly_labels == -1).astype(int)
    featured["anomaly_score"] = anomaly_scores

    season_anomaly_counts = {}
    if "season" in featured.columns:
        for season_label in sorted(pd.unique(featured["season"])):
            mask = featured["season"] == season_label
            season_anomalies = int((anomaly_labels[mask] == -1).sum())
            season_anomaly_counts[f"IF__anomaly_{season_label}"] = season_anomalies
            featured[f"IF__anomaly_{season_label}"] = season_anomalies

    anomaly_count = int((anomaly_labels == -1).sum())
    anomaly_density = float(anomaly_count / len(anomaly_labels)) if len(anomaly_labels) else np.nan
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)
    anomaly_mean_score = float(np.nanmean(anomaly_scores)) if anomaly_scores.size else np.nan
    anomaly_min_score = float(np.nanmin(anomaly_scores)) if anomaly_scores.size else np.nan

    metrics = pd.DataFrame(
        {
            "Persistence": evaluate_research_metrics(y_test.values, persistence_pred, model_name="Persistence", district=district),
            "Random Forest": evaluate_research_metrics(y_test.values, rf_pred, model_name="Random Forest", district=district),
            "XGBoost": evaluate_research_metrics(y_test.values, xgb_pred, model_name="XGBoost", district=district),
        }
    ).T

    metrics["Regime Accuracy"] = [persistence_regime_accuracy, rf_regime_accuracy, xgb_regime_accuracy]
    metrics["Regime Acc Low"] = [
        float((actual_regime[actual_regime == "Low"] == persistence_regime[actual_regime == "Low"]).mean()) if actual_regime_counts["Low"] > 0 else np.nan,
        float((actual_regime[actual_regime == "Low"] == rf_regime[actual_regime == "Low"]).mean()) if actual_regime_counts["Low"] > 0 else np.nan,
        float((actual_regime[actual_regime == "Low"] == xgb_regime[actual_regime == "Low"]).mean()) if actual_regime_counts["Low"] > 0 else np.nan,
    ]
    metrics["Regime Acc Moderate"] = [
        float((actual_regime[actual_regime == "Moderate"] == persistence_regime[actual_regime == "Moderate"]).mean()) if actual_regime_counts["Moderate"] > 0 else np.nan,
        float((actual_regime[actual_regime == "Moderate"] == rf_regime[actual_regime == "Moderate"]).mean()) if actual_regime_counts["Moderate"] > 0 else np.nan,
        float((actual_regime[actual_regime == "Moderate"] == xgb_regime[actual_regime == "Moderate"]).mean()) if actual_regime_counts["Moderate"] > 0 else np.nan,
    ]
    metrics["Regime Acc High"] = [
        float((actual_regime[actual_regime == "High"] == persistence_regime[actual_regime == "High"]).mean()) if actual_regime_counts["High"] > 0 else np.nan,
        float((actual_regime[actual_regime == "High"] == rf_regime[actual_regime == "High"]).mean()) if actual_regime_counts["High"] > 0 else np.nan,
        float((actual_regime[actual_regime == "High"] == xgb_regime[actual_regime == "High"]).mean()) if actual_regime_counts["High"] > 0 else np.nan,
    ]

    if regime_class_count < 3:
        regime_eval_note = f"Regime accuracy is shown, but only {regime_class_count} regime class(es) appear in the test set, so this score is weak evidence for cross-model comparison."
    else:
        regime_eval_note = "Regime accuracy is computed across all three classes in the hold-out set and is suitable for model comparison."
    metrics["Regime Coverage"] = float((actual_regime_counts > 0).sum()) / float(len(regime_labels))
    metrics["Regime Majority Share"] = float(actual_regime_counts.max() / max(actual_regime_counts.sum(), 1))
    metrics["Regime Valid"] = float(regime_class_count == 3)
    metrics["Regime Weight"] = metrics["Regime Valid"]

    try:
        import shap
        for model_name, model_obj in [("Random Forest", rf), ("XGBoost", xgb)]:
            explainer = shap.TreeExplainer(model_obj)
            shap_vals = explainer.shap_values(X_test)
            mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_test.columns)
            for feat in X_test.columns:
                metrics[f"SHAP__{model_name}__{feat}"] = float(mean_shap[feat])
            top3 = mean_shap.nlargest(3)
            for rank, (feat, val) in enumerate(top3.items(), 1):
                metrics[f"SHAP__{model_name}__top{rank}_feature"] = feat
                metrics[f"SHAP__{model_name}__top{rank}_value"] = float(val)
    except Exception as exc:
        print(f"      WARNING: SHAP values not computed for {district or 'district'}: {exc}")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "rf": rf,
        "xgb": xgb,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "persistence_pred": persistence_pred,
        "metrics": metrics,
        "featured": featured,
        "feature_cols": feature_cols,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "regime_bins": regime_bins,
        "regime_labels": regime_labels,
        "actual_regime_counts": actual_regime_counts,
        "regime_eval_note": regime_eval_note,
        "split_ratio": 0.80,
        "regime_valid": regime_class_count == 3,
        "regime_class_count": regime_class_count,
        "regime_confusion_matrices": {
            "Persistence": persistence_cm,
            "Random Forest": rf_cm,
            "XGBoost": xgb_cm,
        },
        "IF__anomaly_count": anomaly_count,
        "IF__total_count": int(len(anomaly_labels)),
        "IF__anomaly_density": anomaly_density,
        "IF__mean_score": anomaly_mean_score,
        "IF__min_score": anomaly_min_score,
        "IF__season_anomaly_counts": season_anomaly_counts,
    }


def flatten_metrics_row(state, district, agency, trained):
    metrics = trained["metrics"]
    row = {
        "state": state,
        "district": district,
        "agency": agency,
        "rows": len(trained["featured"]),
        "train_rows": len(trained["train_df"]),
        "test_rows": len(trained["test_df"]),
        "regime_valid": bool(trained["regime_valid"]),
        "regime_class_count": int(trained["regime_class_count"]),
        "regime_eval_note": trained["regime_eval_note"],
        "split_ratio": trained["split_ratio"],
        "Regime Coverage": float(trained["metrics"]["Regime Coverage"].iloc[0]) if "Regime Coverage" in trained["metrics"].columns else np.nan,
        "Regime Majority Share": float(trained["metrics"]["Regime Majority Share"].iloc[0]) if "Regime Majority Share" in trained["metrics"].columns else np.nan,
        "gwl_min": float(trained["metrics"]["gwl_min"].iloc[0]) if "gwl_min" in trained["metrics"].columns else np.nan,
        "gwl_max": float(trained["metrics"]["gwl_max"].iloc[0]) if "gwl_max" in trained["metrics"].columns else np.nan,
        "gwl_range": float(trained["metrics"]["gwl_range"].iloc[0]) if "gwl_range" in trained["metrics"].columns else np.nan,
        "IF__anomaly_count": int(trained.get("IF__anomaly_count", np.nan)) if trained.get("IF__anomaly_count", None) is not None else np.nan,
        "IF__total_count": int(trained.get("IF__total_count", np.nan)) if trained.get("IF__total_count", None) is not None else np.nan,
        "IF__anomaly_density": float(trained.get("IF__anomaly_density", np.nan)) if trained.get("IF__anomaly_density", None) is not None else np.nan,
        "IF__mean_score": float(trained.get("IF__mean_score", np.nan)) if trained.get("IF__mean_score", None) is not None else np.nan,
        "IF__min_score": float(trained.get("IF__min_score", np.nan)) if trained.get("IF__min_score", None) is not None else np.nan,
    }

    for model_name in metrics.index.tolist():
        for col_name in metrics.columns:
            value = metrics.loc[model_name, col_name]
            if isinstance(col_name, str) and col_name.startswith("SHAP__"):
                row[col_name] = value
            else:
                row[f"{model_name}__{col_name}"] = value

    for model_name in ["Persistence", "Random Forest", "XGBoost"]:
        cm = trained["regime_confusion_matrices"][model_name]
        for actual in cm.index:
            for pred in cm.columns:
                row[f"{model_name}_cm__{actual}__{pred}"] = int(cm.loc[actual, pred])

    for season_col, season_count in trained.get("IF__season_anomaly_counts", {}).items():
        row[season_col] = int(season_count)

    return row


MODEL_NAMES = ["Persistence", "Random Forest", "XGBoost"]
METRIC_NAMES = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "NSE",
    "NRMSE",
    "PBIAS",
    "Pearson r",
    "Bias",
    "KGE",
    "Alpha",
    "Beta",
    "Regime Accuracy",
    "Regime Acc Low",
    "Regime Acc Moderate",
    "Regime Acc High",
]


APPEND_COLUMNS = [
    "state", "district", "agency", "rows", "train_rows", "test_rows", "regime_valid",
    "regime_class_count", "regime_eval_note", "split_ratio", "Regime Coverage", "Regime Majority Share",
    "gwl_min", "gwl_max", "gwl_range",
    "IF__anomaly_count", "IF__total_count", "IF__anomaly_density", "IF__mean_score", "IF__min_score",
] + [f"{model}__{metric}" for model in MODEL_NAMES for metric in METRIC_NAMES] + [
    f"{model}_cm__Actual {actual}__Pred {pred}"
    for model in MODEL_NAMES
    for actual in ["Low", "Moderate", "High"]
    for pred in ["Low", "Moderate", "High"]
]


def ensure_csv_header(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def build_empty_state_row(state, district, agency):
    row = {
        "state": state,
        "district": district or "N/A",
        "agency": agency,
        "rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "regime_valid": False,
        "regime_class_count": 0,
        "regime_eval_note": "No backend data available for this state/district.",
        "split_ratio": 0.0,
        "Regime Coverage": 0.0,
        "Regime Majority Share": 0.0,
    }
    for model_name in MODEL_NAMES:
        for metric_name in METRIC_NAMES:
            row[f"{model_name}__{metric_name}"] = 0.0
    for model_name in MODEL_NAMES:
        for actual in ["Actual Low", "Actual Moderate", "Actual High"]:
            for pred in ["Pred Low", "Pred Moderate", "Pred High"]:
                row[f"{model_name}_cm__{actual}__{pred}"] = 0
    return row


def choose_agency(agencies, requested_agency=None):
    if requested_agency:
        if requested_agency in agencies:
            return requested_agency
        print(f"Warning: requested agency '{requested_agency}' not found; using the first available agency instead.")
    if "CGWB" in agencies:
        return "CGWB"
    return agencies[0] if agencies else None


def main():
    args = parse_args()
    session = requests.Session()
    session.verify = True

    output_path = os.path.abspath(args.output)
    append_path = os.path.abspath(args.append_output) if args.append_output else None
    if append_path:
        print(f"Appending results to: {append_path}")

    states = [args.state] if args.state else fetch_states(session, args.backend_url)
    if not states:
        print("No states found. Check that the backend is running and accessible.")
        return

    agencies = fetch_agencies(session, args.backend_url)
    agency = choose_agency(agencies, args.agency)
    if agency is None:
        print("No agency available from the backend. Check /agencies endpoint.")
        return

    print(f"Using backend: {args.backend_url}")
    print(f"Using agency: {agency}")
    print(f"States to process: {len(states)}")

    results = []
    for state in states:
        if args.district:
            districts = [args.district]
        else:
            try:
                districts = fetch_districts(session, args.backend_url, state)
            except Exception as exc:
                print(f"  WARNING: failed to load districts for state {state}: {exc}")
                districts = []

        if not districts:
            print(f"  No district list for state {state}. Writing zero-filled row for analysis.")
            results.append(build_empty_state_row(state, args.district or "N/A", agency))
            continue

        print(f"Processing state: {state} with {len(districts)} districts")
        state_has_result = False
        for district in districts:
            print(f"    Fetching {state} / {district}")
            try:
                raw_df = fetch_groundwater_data(
                    session,
                    args.backend_url,
                    state,
                    district,
                    agency,
                    args.start_date,
                    args.end_date,
                    args.size,
                )
            except Exception as exc:
                print(f"      ERROR: failed to fetch data for {state}/{district}: {exc}")
                continue

            featured = normalize_groundwater_frame(raw_df)
            if featured is None or featured.empty:
                print(f"      WARNING: no valid groundwater rows for {state}/{district}")
                continue

            featured, feature_cols = build_features(featured)
            trained = train_models(featured, feature_cols, district=district)
            if trained is None:
                print(f"      WARNING: not enough data after feature engineering for {state}/{district}")
                continue

            row = flatten_metrics_row(state, district, agency, trained)
            results.append(row)
            state_has_result = True

        if not state_has_result:
            print(f"  No successful district metrics for state {state}. Adding zero-filled row.")
            row = build_empty_state_row(state, args.district or "N/A", agency)
            results.append(row)
            if append_path:
                pd.DataFrame([row]).to_csv(append_path, mode="a", header=False, index=False)

    if not results:
        print("No metrics were generated. Exiting.")
        append_path = args.append_output
        if append_path:
            append_path = os.path.abspath(append_path)
            if not os.path.exists(append_path):
                    pd.DataFrame([], columns=APPEND_COLUMNS).to_csv(append_path, index=False)
    print(f"Saved research metrics to {output_path}")

    append_path = args.append_output
    if append_path:
        append_path = os.path.abspath(append_path)
        if os.path.exists(append_path):
            existing_cols = pd.read_csv(append_path, nrows=0).columns.tolist()
            if existing_cols != df.columns.tolist():
                print("WARNING: append CSV columns differ from current metrics output; rewriting append file with full current schema.")
                df.to_csv(append_path, index=False)
            else:
                df.to_csv(append_path, mode="a", header=False, index=False)
        else:
            df.to_csv(append_path, index=False)
        print(f"Appended research metrics to {append_path}")

    try:
        from benchmark_utils import build_pooled_summaries

        pooled = build_pooled_summaries(df)
        if pooled is not None:
            print("\nPooled summary metrics by model:")
            print(pooled["summary"].to_string(float_format="{:.4f}".format))
            print("\nAggregated XGBoost confusion matrix:")
            print(pooled["xgb_cm"].to_string())
    except Exception as exc:
        print(f"Could not build pooled summary: {exc}")


if __name__ == "__main__":
    main()
