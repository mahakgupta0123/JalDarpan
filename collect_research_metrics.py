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

# ─────────────────────────────────────────────────────────────────────────────
# JalDarpan — Research Metrics Sweep
# Fixed version — all critical bugs and warnings resolved:
#
#  FIX 1: df was never assigned in main() — NameError crash on CSV save
#  FIX 2: MAPE used np.maximum(..., 1e-6) clamp — inflated errors near zero
#  FIX 3: XGBoost eval_set received fake zeros via walrus operator —
#          corrupted early stopping signal
#  FIX 4: NSE was mathematically identical to R² — now uses modified NSE
#          (mNSE) with absolute residuals so it genuinely differs from R²
#  FIX 5: KGE was NaN for Persistence (std=0 guard too aggressive) —
#          now computes correctly for constant predictors
#  FIX 6: Regime accuracy averaged over districts including invalid ones
#          (fewer than 3 classes) — now filtered before aggregation
#  FIX 7: build_empty_state_row() filled metrics with 0.0 instead of NaN —
#          zeros pulled national averages toward "perfect prediction"
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
DEFAULT_OUTPUT = os.environ.get("RESEARCH_SWEEP_CSV", "research_metrics_sweep.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect research metrics for all states and districts."
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="URL of the Flask backend (example: http://localhost:5000)",
    )
    parser.add_argument(
        "--agency",
        default=None,
        help="Agency to use for all district queries. If omitted, uses CGWB or the first agency returned.",
    )
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date for groundwater data retrieval.",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for groundwater data retrieval.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Max records to request per district.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV file to write the collected metrics.",
    )
    parser.add_argument(
        "--append-output",
        default=os.environ.get("RESEARCH_APPEND_CSV", "research_metrics_append.csv"),
        help="CSV file to append collected metrics for longer-term accumulation.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional single state to run instead of all states.",
    )
    parser.add_argument(
        "--district",
        default=None,
        help="Optional single district to run instead of all districts in a state.",
    )
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


def fetch_groundwater_data(
    session, backend_url, state, district, agency, startdate, enddate, size=500
):
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

    time_col = _safe_column(
        df, ["dataTime", "date", "timestamp", "recordDate"]
    )
    value_col = _safe_column(
        df, ["dataValue", "value", "waterLevel", "groundwater_level"]
    )

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
    data["rolling_7_std"] = (
        data["groundwater_level"].rolling(7, min_periods=1).std().fillna(0)
    )
    data["rolling_14_std"] = (
        data["groundwater_level"].rolling(14, min_periods=1).std().fillna(0)
    )
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


def compute_mape(y_true, y_pred):
    """
    FIX 2: Standard MAPE excluding near-zero true values.
    Original used np.maximum(|y_true|, 1e-6) which inflated errors
    for any shallow-water reading near zero.
    Threshold of 0.01m is safe for groundwater depth (always > 0 in practice).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 0.01
    if not mask.any():
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def compute_mnse(y_true, y_pred):
    """
    FIX 4: Modified NSE using absolute residuals instead of squared.
    Standard NSE = 1 - sum(e²)/sum((obs - mean_obs)²) is mathematically
    identical to R². mNSE = 1 - sum(|e|)/sum(|obs - mean_obs|) is a
    genuinely different metric, more robust to outliers, and will produce
    different values from R² in every district.
    Reference: Krause et al. (2005), Advances in Geosciences.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(1.0 - np.sum(np.abs(y_true - y_pred)) / denom)


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

    # FIX 2: corrected MAPE
    mape = compute_mape(y_true, y_pred)

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan

    # FIX 4: mNSE genuinely different from R²
    mnse = compute_mnse(y_true, y_pred)

    if gwl_range > 0:
        nrmse = float(rmse / gwl_range)
        if nrmse > 1.0:
            tag = f" for district '{district}'" if district else ""
            tag2 = f" ({model_name})" if model_name else ""
            print(f"      WARNING: NRMSE{tag2}{tag} = {nrmse:.4f} > 1.0")
    else:
        nrmse = np.nan

    pbias = float(
        100.0 * np.sum(y_pred - y_true) / np.maximum(np.sum(y_true), 1e-6)
    )
    bias = float(np.mean(y_pred - y_true))

    # ── KGE and its components (FIX 5) ───────────────────────────────────────
    # Compute std once — used for both Pearson r and alpha
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    is_constant_pred = np.isclose(std_pred, 0.0)

    if len(y_true) > 1 and not np.isclose(std_true, 0.0):
        if is_constant_pred:
            # Constant predictor (e.g. Persistence):
            # r=0 by definition (no co-variation), alpha=0 (no variability)
            pearson_r = 0.0
            alpha = 0.0
        else:
            pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
            alpha = std_pred / std_true

        if mean_true != 0:
            beta = float(np.mean(y_pred) / mean_true)
            if abs(beta) > 10:
                # Degenerate district — mean_true ≈ 0 causes beta overflow
                kge = np.nan
            else:
                kge = float(
                    1.0 - np.sqrt(
                        (pearson_r - 1.0) ** 2
                        + (alpha - 1.0) ** 2
                        + (beta - 1.0) ** 2
                    )
                )
        else:
            alpha = np.nan
            beta = np.nan
            kge = np.nan
    else:
        pearson_r = np.nan
        alpha = np.nan
        beta = np.nan
        kge = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "mNSE": mnse,
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
    return pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Pred {label}" for label in labels],
    )


def _fit_tuned_xgboost(X_train, y_train, X_val, y_val):
    candidate_params = [
        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "gamma": 0.0,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 2,
            "gamma": 0.1,
        },
        {
            "n_estimators": 800,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "min_child_weight": 1,
            "gamma": 0.2,
        },
        {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.3,
            "reg_lambda": 2.5,
            "min_child_weight": 1,
            "gamma": 0.3,
        },
    ]

    best_model = None
    best_rmse = np.inf
    best_params = None

    for params in candidate_params:
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **params,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
        except Exception:
            model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    if best_model is None:
        best_model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.0,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)

    final_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **(best_params or {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "gamma": 0.0,
        }),
    )
    final_model.fit(X_train, y_train)
    return final_model


def train_models(featured, feature_cols, district=None):
    if len(featured) < 20:
        return None

    train_df, test_df = time_series_split(featured)
    if test_df.empty:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df["groundwater_level"]
    X_test = test_df[feature_cols]

    y_test = test_df["groundwater_level"].values
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

    if len(train_df) >= 24:
        split_idx = max(8, int(len(train_df) * 0.8))
        X_fit = X_train.iloc[:split_idx]
        y_fit = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        xgb_model = _fit_tuned_xgboost(X_fit, y_fit, X_val, y_val)
    else:
        xgb_model = _fit_tuned_xgboost(X_train, y_train, X_train.iloc[:max(4, len(X_train) // 2)], y_train.iloc[:max(4, len(y_train) // 2)])

    xgb_pred = xgb_model.predict(X_test)

    # Persistence: predict last training value for all test steps
    persistence_pred = np.repeat(float(y_train.iloc[-1]), len(y_test))

    # ── Regime classification ─────────────────────────────────────────────────
    regime_bins, regime_labels = build_regime_bins(y_train.values)
    actual_regime = make_regime_labels(y_test, regime_bins, regime_labels)
    persistence_regime = make_regime_labels(persistence_pred, regime_bins, regime_labels)
    rf_regime = make_regime_labels(rf_pred, regime_bins, regime_labels)
    xgb_regime = make_regime_labels(xgb_pred, regime_bins, regime_labels)

    actual_regime_counts = (
        actual_regime.value_counts(dropna=False).reindex(regime_labels, fill_value=0)
    )
    regime_class_count = int((actual_regime_counts > 0).sum())

    # FIX 6: regime accuracy is only meaningful when all 3 classes are present.
    # We still compute it here so it's stored per district, but the aggregation
    # in main() must filter to regime_valid == True rows before averaging.
    persistence_regime_accuracy = float((actual_regime == persistence_regime).mean())
    rf_regime_accuracy = float((actual_regime == rf_regime).mean())
    xgb_regime_accuracy = float((actual_regime == xgb_regime).mean())

    persistence_cm = regime_confusion_matrix(
        y_test, persistence_pred, regime_bins, regime_labels
    )
    rf_cm = regime_confusion_matrix(y_test, rf_pred, regime_bins, regime_labels)
    xgb_cm = regime_confusion_matrix(y_test, xgb_pred, regime_bins, regime_labels)

    # Residual-based confidence interval for XGBoost predictions
    residuals_xgb = y_test - xgb_pred
    interval_low = float(np.quantile(residuals_xgb, 0.1))
    interval_high = float(np.quantile(residuals_xgb, 0.9))

    # ── Isolation Forest anomaly detection ───────────────────────────────────
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(X_train)
    anomaly_labels = anomaly_detector.predict(X_full)
    anomaly_scores = anomaly_detector.decision_function(X_full)

    featured = featured.copy()
    featured["anomaly_flag"] = (anomaly_labels == -1).astype(int)
    featured["anomaly_score"] = anomaly_scores

    season_anomaly_counts = {}
    if "season" in featured.columns:
        for season_label in sorted(pd.unique(featured["season"])):
            mask = featured["season"] == season_label
            season_count = int((anomaly_labels[mask] == -1).sum())
            season_anomaly_counts[f"IF__anomaly_{season_label}"] = season_count
            featured[f"IF__anomaly_{season_label}"] = season_count

    anomaly_count = int((anomaly_labels == -1).sum())
    anomaly_density = (
        float(anomaly_count / len(anomaly_labels)) if len(anomaly_labels) else np.nan
    )
    anomaly_scores_arr = np.asarray(anomaly_scores, dtype=float)
    anomaly_mean_score = (
        float(np.nanmean(anomaly_scores_arr)) if anomaly_scores_arr.size else np.nan
    )
    anomaly_min_score = (
        float(np.nanmin(anomaly_scores_arr)) if anomaly_scores_arr.size else np.nan
    )

    # ── Per-class regime accuracy (safe: returns NaN if class absent) ────────
    def safe_class_acc(actual, predicted, class_label, counts):
        if counts[class_label] == 0:
            return np.nan
        mask = actual == class_label
        return float((actual[mask] == predicted[mask]).mean())

    # ── Build metrics DataFrame ───────────────────────────────────────────────
    metrics = pd.DataFrame(
        {
            "Persistence": evaluate_research_metrics(
                y_test, persistence_pred, model_name="Persistence", district=district
            ),
            "Random Forest": evaluate_research_metrics(
                y_test, rf_pred, model_name="Random Forest", district=district
            ),
            "XGBoost": evaluate_research_metrics(
                y_test, xgb_pred, model_name="XGBoost", district=district
            ),
        }
    ).T

    metrics["Regime Accuracy"] = [
        persistence_regime_accuracy,
        rf_regime_accuracy,
        xgb_regime_accuracy,
    ]
    # FIX 6: per-class accuracy uses NaN when that class is absent in test set
    metrics["Regime Acc Low"] = [
        safe_class_acc(actual_regime, persistence_regime, "Low", actual_regime_counts),
        safe_class_acc(actual_regime, rf_regime, "Low", actual_regime_counts),
        safe_class_acc(actual_regime, xgb_regime, "Low", actual_regime_counts),
    ]
    metrics["Regime Acc Moderate"] = [
        safe_class_acc(actual_regime, persistence_regime, "Moderate", actual_regime_counts),
        safe_class_acc(actual_regime, rf_regime, "Moderate", actual_regime_counts),
        safe_class_acc(actual_regime, xgb_regime, "Moderate", actual_regime_counts),
    ]
    metrics["Regime Acc High"] = [
        safe_class_acc(actual_regime, persistence_regime, "High", actual_regime_counts),
        safe_class_acc(actual_regime, rf_regime, "High", actual_regime_counts),
        safe_class_acc(actual_regime, xgb_regime, "High", actual_regime_counts),
    ]

    if regime_class_count < 3:
        regime_eval_note = (
            f"Regime accuracy shown but only {regime_class_count} class(es) present "
            f"in test set — exclude from cross-district regime comparison."
        )
    else:
        regime_eval_note = (
            "Regime accuracy computed across all 3 classes in hold-out set "
            "— suitable for model comparison."
        )

    metrics["Regime Coverage"] = float((actual_regime_counts > 0).sum()) / float(
        len(regime_labels)
    )
    metrics["Regime Majority Share"] = float(
        actual_regime_counts.max() / max(actual_regime_counts.sum(), 1)
    )
    metrics["Regime Valid"] = float(regime_class_count == 3)
    # FIX 6: Regime Weight stored as boolean flag; aggregation in main()
    # must filter rows where Regime Valid == 1.0 before computing averages.
    metrics["Regime Weight"] = metrics["Regime Valid"]

    # ── SHAP values ───────────────────────────────────────────────────────────
    try:
        import shap

        for model_name_shap, model_obj in [("Random Forest", rf), ("XGBoost", xgb_model)]:
            explainer = shap.TreeExplainer(model_obj)
            shap_vals = explainer.shap_values(X_test)
            mean_shap = pd.Series(
                np.abs(shap_vals).mean(axis=0), index=X_test.columns
            )
            for feat in X_test.columns:
                metrics[f"SHAP__{model_name_shap}__{feat}"] = float(mean_shap[feat])
            top3 = mean_shap.nlargest(3)
            for rank, (feat, val) in enumerate(top3.items(), 1):
                metrics[f"SHAP__{model_name_shap}__top{rank}_feature"] = feat
                metrics[f"SHAP__{model_name_shap}__top{rank}_value"] = float(val)
    except Exception as exc:
        print(
            f"      WARNING: SHAP values not computed for {district or 'district'}: {exc}"
        )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "rf": rf,
        "xgb": xgb_model,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "y_test": y_test,
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
        "Regime Coverage": (
            float(trained["metrics"]["Regime Coverage"].iloc[0])
            if "Regime Coverage" in trained["metrics"].columns
            else np.nan
        ),
        "Regime Majority Share": (
            float(trained["metrics"]["Regime Majority Share"].iloc[0])
            if "Regime Majority Share" in trained["metrics"].columns
            else np.nan
        ),
        "gwl_min": (
            float(trained["metrics"]["gwl_min"].iloc[0])
            if "gwl_min" in trained["metrics"].columns
            else np.nan
        ),
        "gwl_max": (
            float(trained["metrics"]["gwl_max"].iloc[0])
            if "gwl_max" in trained["metrics"].columns
            else np.nan
        ),
        "gwl_range": (
            float(trained["metrics"]["gwl_range"].iloc[0])
            if "gwl_range" in trained["metrics"].columns
            else np.nan
        ),
        "IF__anomaly_count": (
            int(trained.get("IF__anomaly_count"))
            if trained.get("IF__anomaly_count") is not None
            else np.nan
        ),
        "IF__total_count": (
            int(trained.get("IF__total_count"))
            if trained.get("IF__total_count") is not None
            else np.nan
        ),
        "IF__anomaly_density": (
            float(trained.get("IF__anomaly_density"))
            if trained.get("IF__anomaly_density") is not None
            else np.nan
        ),
        "IF__mean_score": (
            float(trained.get("IF__mean_score"))
            if trained.get("IF__mean_score") is not None
            else np.nan
        ),
        "IF__min_score": (
            float(trained.get("IF__min_score"))
            if trained.get("IF__min_score") is not None
            else np.nan
        ),
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
            for pred_col in cm.columns:
                row[f"{model_name}_cm__{actual}__{pred_col}"] = int(
                    cm.loc[actual, pred_col]
                )

    for season_col, season_count in trained.get("IF__season_anomaly_counts", {}).items():
        row[season_col] = int(season_count)

    return row


MODEL_NAMES = ["Persistence", "Random Forest", "XGBoost"]

# FIX 4: NSE renamed to mNSE throughout
METRIC_NAMES = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2",
    "mNSE",          # FIX 4: was "NSE" — now Modified NSE (genuinely ≠ R²)
    "NRMSE",
    "PBIAS",
    "Pearson r",
    "Bias",
    "KGE",           # FIX 5: now computed for constant predictors too
    "Alpha",
    "Beta",
    "Regime Accuracy",
    "Regime Acc Low",
    "Regime Acc Moderate",
    "Regime Acc High",
]

APPEND_COLUMNS = (
    [
        "state", "district", "agency", "rows", "train_rows", "test_rows",
        "regime_valid", "regime_class_count", "regime_eval_note", "split_ratio",
        "Regime Coverage", "Regime Majority Share",
        "gwl_min", "gwl_max", "gwl_range",
        "IF__anomaly_count", "IF__total_count", "IF__anomaly_density",
        "IF__mean_score", "IF__min_score",
    ]
    + [f"{model}__{metric}" for model in MODEL_NAMES for metric in METRIC_NAMES]
    + [
        f"{model}_cm__Actual {actual}__Pred {pred}"
        for model in MODEL_NAMES
        for actual in ["Low", "Moderate", "High"]
        for pred in ["Low", "Moderate", "High"]
    ]
)


def build_empty_state_row(state, district, agency):
    """
    FIX 7: Empty rows now use np.nan instead of 0.0 for all metric fields.
    Original used 0.0 which means "perfect prediction" — including those
    rows in any national average pulled reported accuracy toward zero error.
    With np.nan, aggregation using np.nanmean() will skip these rows
    correctly. The rows still count toward the district total for reporting.
    """
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
        "Regime Coverage": np.nan,       # FIX 7: was 0.0
        "Regime Majority Share": np.nan, # FIX 7: was 0.0
        "gwl_min": np.nan,
        "gwl_max": np.nan,
        "gwl_range": np.nan,
        "IF__anomaly_count": np.nan,
        "IF__total_count": np.nan,
        "IF__anomaly_density": np.nan,
        "IF__mean_score": np.nan,
        "IF__min_score": np.nan,
    }
    # FIX 7: all model metrics are NaN, not 0.0
    for model_name in MODEL_NAMES:
        for metric_name in METRIC_NAMES:
            row[f"{model_name}__{metric_name}"] = np.nan
    for model_name in MODEL_NAMES:
        for actual in ["Actual Low", "Actual Moderate", "Actual High"]:
            for pred in ["Pred Low", "Pred Moderate", "Pred High"]:
                row[f"{model_name}_cm__{actual}__{pred}"] = 0
    return row


def choose_agency(agencies, requested_agency=None):
    if requested_agency:
        if requested_agency in agencies:
            return requested_agency
        print(
            f"Warning: requested agency '{requested_agency}' not found; "
            f"using the first available agency instead."
        )
    if "CGWB" in agencies:
        return "CGWB"
    return agencies[0] if agencies else None


def print_step9_preview(df):
    """
    Reproduce the step-9 metrics preview table shown in the original run,
    now with mNSE replacing NSE and KGE computed for all models.
    """
    preview_metrics = ["NRMSE", "R2", "mNSE", "KGE"]
    rows = []
    for model in MODEL_NAMES:
        row_data = {"Model": model}
        for metric in preview_metrics:
            col = f"{model}__{metric}"
            if col in df.columns:
                # FIX 6: regime accuracy aggregation uses only valid districts
                vals = df[col].dropna()
                row_data[f"{metric} (mean)"] = vals.mean()
        valid_count = int((df["rows"] > 0).sum())
        row_data["Valid Districts"] = valid_count
        rows.append(row_data)

    preview_df = pd.DataFrame(rows).set_index("Model")
    print("\nSTEP 9 - Metrics preview")
    print(preview_df.to_string(float_format="{:.6f}".format))

    # FIX 6: also print regime accuracy on valid-only districts
    valid_regime_df = df[df["regime_valid"] == True]
    if not valid_regime_df.empty:
        print(
            f"\nRegime accuracy (computed on {len(valid_regime_df)} districts "
            f"with all 3 classes present, out of {len(df)} total):"
        )
        for model in MODEL_NAMES:
            col = f"{model}__Regime Accuracy"
            if col in valid_regime_df.columns:
                mean_acc = valid_regime_df[col].mean()
                print(f"  {model}: {mean_acc:.4f}")


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
            print(
                f"  No district list for state {state}. "
                f"Writing NaN-filled row for analysis."  # FIX 7: was "zero-filled"
            )
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
                print(
                    f"      ERROR: failed to fetch data for {state}/{district}: {exc}"
                )
                continue

            normalized = normalize_groundwater_frame(raw_df)
            if normalized is None or normalized.empty:
                print(
                    f"      WARNING: no valid groundwater rows for {state}/{district}"
                )
                continue

            featured, feature_cols = build_features(normalized)
            trained = train_models(featured, feature_cols, district=district)
            if trained is None:
                print(
                    f"      WARNING: not enough data after feature engineering "
                    f"for {state}/{district}"
                )
                continue

            row = flatten_metrics_row(state, district, agency, trained)
            results.append(row)
            state_has_result = True

        if not state_has_result:
            print(
                f"  No successful district metrics for state {state}. "
                f"Adding NaN-filled row."  # FIX 7
            )
            empty_row = build_empty_state_row(state, args.district or "N/A", agency)
            results.append(empty_row)

    if not results:
        print("No metrics were generated. Exiting.")
        if append_path:
            append_path_abs = os.path.abspath(append_path)
            if not os.path.exists(append_path_abs):
                pd.DataFrame(columns=APPEND_COLUMNS).to_csv(
                    append_path_abs, index=False
                )
        return

    # ── FIX 1: df was never assigned — NameError crashed the script ───────────
    # Build the DataFrame from results BEFORE any CSV save or column check.
    df = pd.DataFrame(results)

    # Save primary output
    df.to_csv(output_path, index=False)
    print(f"Saved research metrics to {output_path}")

    # ── Append output ─────────────────────────────────────────────────────────
    if append_path:
        append_path_abs = os.path.abspath(append_path)
        if os.path.exists(append_path_abs):
            existing_cols = pd.read_csv(append_path_abs, nrows=0).columns.tolist()
            if existing_cols != df.columns.tolist():
                print(
                    "WARNING: append CSV columns differ from current output; "
                    "attempting to recover existing rows."
                )
                try:
                    existing_df = pd.read_csv(append_path_abs)
                except Exception:
                    # try reading as headerless file in case the prior append created no header
                    existing_df = pd.read_csv(append_path_abs, header=None)
                    if len(existing_df.columns) == len(df.columns):
                        existing_df.columns = df.columns
                    else:
                        raise
                if list(existing_df.columns) != df.columns.tolist():
                    print(
                        "WARNING: could not recover header structure from append CSV; "
                        "rewriting it with the current schema."
                    )
                    df.to_csv(append_path_abs, index=False)
                else:
                    combined = pd.concat([existing_df, df], ignore_index=True, sort=False)
                    combined.to_csv(append_path_abs, index=False)
            else:
                df.to_csv(append_path_abs, mode="a", header=False, index=False)
        else:
            df.to_csv(append_path_abs, index=False)
        print(f"Appended research metrics to {append_path_abs}")

    # ── Step 9 preview (matches original output format + fixes) ───────────────
    print_step9_preview(df)

    # ── Optional pooled summary ───────────────────────────────────────────────
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