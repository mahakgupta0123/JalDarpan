from datetime import datetime, timedelta
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


st.set_page_config(page_title="JalDarpan", page_icon="🌊", layout="wide")

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
BENCHMARK_CACHE = "research_benchmark_runs.csv"


def _safe_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _safe_text(value, fallback="Unknown"):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _format_driver_name(feature_name):
    if feature_name.startswith("lag_"):
        return f"Recent history ({feature_name.replace('_', ' ')})"
    if feature_name.startswith("rolling_"):
        return f"Short-term trend or spread ({feature_name.replace('_', ' ')})"
    if feature_name in {"month", "dayofyear", "weekday", "is_monsoon"}:
        return f"Seasonality ({feature_name.replace('_', ' ')})"
    return feature_name.replace("_", " ").title()


def _plotly_chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


def _geocode_location(state, district):
    query = ", ".join(part for part in [_safe_text(district, ""), _safe_text(state, ""), "India"] if part)
    if not query.strip():
        return None

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "JalDarpan/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        first = payload[0]
        return {
            "lat": float(first["lat"]),
            "lon": float(first["lon"]),
            "label": first.get("display_name", query),
            "source": "geocoded district/state",
        }
    except Exception:
        return None


def build_location_context(raw_df, state, district):
    lat_col = _safe_column(raw_df, ["latitude", "lat", "Latitude", "Lat", "y"])
    lon_col = _safe_column(raw_df, ["longitude", "lon", "Longitude", "Long", "lng", "x"])
    value_col = _safe_column(raw_df, ["dataValue", "value", "waterLevel", "groundwater_level"])
    label_col = _safe_column(raw_df, ["stationName", "station_name", "location", "site", "siteName", "wellName"])

    if lat_col and lon_col:
        columns = [lat_col, lon_col]
        if value_col:
            columns.append(value_col)
        geo = raw_df[columns].copy()
        geo[lat_col] = pd.to_numeric(geo[lat_col], errors="coerce")
        geo[lon_col] = pd.to_numeric(geo[lon_col], errors="coerce")
        if value_col:
            geo[value_col] = pd.to_numeric(geo[value_col], errors="coerce")
        geo = geo.dropna(subset=[lat_col, lon_col])
        if not geo.empty:
            if label_col:
                geo["label"] = raw_df.loc[geo.index, label_col].fillna("Observation").astype(str)
            else:
                geo["label"] = "Observation"
            if value_col and value_col in geo.columns:
                geo = geo.rename(columns={value_col: "groundwater_level"})
            return {
                "points": geo.rename(columns={lat_col: "lat", lon_col: "lon"}),
                "map_label": f"{_safe_text(district)} • {_safe_text(state)}",
                "source": "raw data coordinates",
            }

    geocoded = _geocode_location(state, district)
    if geocoded is None:
        return None

    return {
        "points": pd.DataFrame(
            [
                {
                    "lat": geocoded["lat"],
                    "lon": geocoded["lon"],
                    "label": f"{_safe_text(district)} district center",
                }
            ]
        ),
        "map_label": geocoded["label"],
        "source": geocoded["source"],
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_groundwater_data(backend_url, state, district, agency, startdate, enddate, size=500):
    params = {
        "state": state,
        "district": district,
        "agency": agency,
        "startdate": startdate,
        "enddate": enddate,
        "size": size,
    }
    response = requests.get(f"{backend_url.rstrip('/')}/groundwater", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise ValueError(payload.get("message", "Unexpected backend response"))

    data = payload.get("data", [])
    if not data:
        raise ValueError("Backend returned no rows")

    return pd.DataFrame(data)


def load_cached_data():
    cached_file = "cached_groundwater_data.csv"
    if os.path.exists(cached_file):
        return pd.read_csv(cached_file)
    return None


def load_benchmark_runs():
    if os.path.exists(BENCHMARK_CACHE):
        return pd.read_csv(BENCHMARK_CACHE)
    return pd.DataFrame()


def record_benchmark_run(meta, metrics, trained):
    row = {
        "state": meta.get("state", ""),
        "district": meta.get("district", ""),
        "agency": meta.get("agency", ""),
        "rows": int(meta.get("rows", 0)),
    }

    for model_name in metrics.index:
        for metric_name in metrics.columns:
            row[f"{model_name}__{metric_name}"] = float(metrics.loc[model_name, metric_name])

    xgb_cm = trained["regime_confusion_matrices"]["XGBoost"]
    for actual_label in xgb_cm.index:
        for pred_label in xgb_cm.columns:
            key = f"xgb_cm__{actual_label}__{pred_label}"
            row[key] = int(xgb_cm.loc[actual_label, pred_label])

    history = load_benchmark_runs()
    updated = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(BENCHMARK_CACHE, index=False)


def build_benchmark_summary(exclude_latest=False):
    runs = load_benchmark_runs()
    if runs.empty:
        return None

    if exclude_latest and len(runs) > 1:
        runs = runs.iloc[:-1].copy()

    summary = {}
    model_names = ["Persistence", "Random Forest", "XGBoost"]
    for model_name in model_names:
        summary[model_name] = {}
        for column_name in runs.columns:
            if column_name.startswith(f"{model_name}__"):
                metric_name = column_name.split("__", 1)[1]
                summary[model_name][metric_name] = float(runs[column_name].median())

    xgb_cm = pd.DataFrame(0, index=["Actual Low", "Actual Moderate", "Actual High"], columns=["Pred Low", "Pred Moderate", "Pred High"])
    for actual_label in xgb_cm.index:
        for pred_label in xgb_cm.columns:
            col_name = f"xgb_cm__{actual_label}__{pred_label}"
            if col_name in runs.columns:
                xgb_cm.loc[actual_label, pred_label] = int(runs[col_name].fillna(0).sum())

    return {
        "runs": runs,
        "summary": pd.DataFrame(summary).T,
        "xgb_cm": xgb_cm,
    }


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

    feature_cols = [
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

    data = data.dropna(subset=["lag_14", "diff_7"])
    return data, feature_cols


def time_series_split(frame, train_ratio=0.8):
    split_index = max(1, int(len(frame) * train_ratio))
    return frame.iloc[:split_index], frame.iloc[split_index:]


def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100),
    }


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


def correlation_heatmap_frame(featured, feature_cols):
    corr_source = featured[["groundwater_level"] + feature_cols].copy()
    return corr_source.corr(numeric_only=True).round(2)


def build_location_map(location_context, featured):
    if location_context is None or location_context.get("points") is None or location_context["points"].empty:
        return None

    points = location_context["points"].copy()
    if "groundwater_level" not in points.columns:
        points["groundwater_level"] = float(featured["groundwater_level"].iloc[-1])

    center_lat = float(points["lat"].mean())
    center_lon = float(points["lon"].mean())

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=points["lat"],
            lon=points["lon"],
            mode="markers",
            marker=dict(
                size=14,
                color=points["groundwater_level"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Level"),
                opacity=0.9,
            ),
            text=points["label"],
            hovertemplate="%{text}<br>Latitude: %{lat:.4f}<br>Longitude: %{lon:.4f}<extra></extra>",
            name="Location",
        )
    )
    fig.update_layout(
        map={"style": "open-street-map", "center": {"lat": center_lat, "lon": center_lon}, "zoom": 7},
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    return fig


def build_anomaly_chart(featured):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=featured.index,
            y=featured["anomaly_score"],
            name="Anomaly score",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#7f7f7f")
    anomaly_points = featured[featured["anomaly_flag"] == 1]
    if not anomaly_points.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points["anomaly_score"],
                mode="markers",
                name="Flagged anomaly",
                marker=dict(color="#d62728", size=8, symbol="x"),
            )
        )
    fig.update_layout(
        height=380,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    return fig


def build_correlation_heatmap(corr_frame):
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_frame.values,
            x=corr_frame.columns.tolist(),
            y=corr_frame.index.tolist(),
            colorscale="YlGnBu",
            zmid=0,
            text=corr_frame.values,
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    return fig


def build_shap_heatmap(sample, shap_values, feature_cols):
    heatmap_rows = min(25, len(sample))
    if heatmap_rows == 0:
        return None

    heat_df = pd.DataFrame(
        shap_values[-heatmap_rows:],
        index=[idx.strftime("%Y-%m-%d") for idx in sample.index[-heatmap_rows:]],
        columns=feature_cols,
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Date: %{y}<br>Feature: %{x}<br>SHAP: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    return fig


def train_models(featured, feature_cols):
    if len(featured) < 20:
        return None

    train_df, test_df = time_series_split(featured)
    if test_df.empty:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df["groundwater_level"]
    X_test = test_df[feature_cols]
    y_test = test_df["groundwater_level"]

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
    actual_regime_labels = make_regime_labels(y_test.values, regime_bins, regime_labels)
    actual_regime_counts = actual_regime_labels.value_counts(dropna=False).reindex(regime_labels, fill_value=0)
    regime_class_count = int((actual_regime_counts > 0).sum())

    persistence_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(persistence_pred, regime_bins, regime_labels)).mean())
    rf_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(rf_pred, regime_bins, regime_labels)).mean())
    xgb_regime_accuracy = float((make_regime_labels(y_test.values, regime_bins, regime_labels) == make_regime_labels(xgb_pred, regime_bins, regime_labels)).mean())

    persistence_cm = regime_confusion_matrix(y_test.values, persistence_pred, regime_bins, regime_labels)
    rf_cm = regime_confusion_matrix(y_test.values, rf_pred, regime_bins, regime_labels)
    xgb_cm = regime_confusion_matrix(y_test.values, xgb_pred, regime_bins, regime_labels)

    residuals = y_test.values - xgb_pred
    interval_low = float(np.quantile(residuals, 0.1))
    interval_high = float(np.quantile(residuals, 0.9))

    anomalies = IsolationForest(contamination=0.05, random_state=42)
    anomaly_input = featured[["groundwater_level", "rolling_7", "rolling_14", "diff_1", "diff_7"]].copy()
    featured = featured.copy()
    featured["anomaly_flag"] = anomalies.fit_predict(anomaly_input)
    featured["anomaly_flag"] = (featured["anomaly_flag"] == -1).astype(int)
    featured["anomaly_score"] = anomalies.decision_function(anomaly_input)

    metrics = pd.DataFrame(
        {
            "Persistence": evaluate_research_metrics(y_test.values, persistence_pred),
            "Random Forest": evaluate_research_metrics(y_test.values, rf_pred),
            "XGBoost": evaluate_research_metrics(y_test.values, xgb_pred),
        }
    ).T

    metrics["Regime Acc"] = [persistence_regime_accuracy, rf_regime_accuracy, xgb_regime_accuracy]
    metrics["Regime Acc"] = metrics["Regime Acc"].astype(float)
    if regime_class_count < 3:
        regime_eval_note = f"Regime accuracy is shown, but only {regime_class_count} regime class(es) appear in the test set, so this score is weak evidence for cross-model comparison."
    else:
        regime_eval_note = "Regime accuracy is computed across all three classes in the hold-out set and is suitable for model comparison."
    metrics["Regime Coverage"] = float((actual_regime_counts > 0).sum()) / float(len(regime_labels))
    metrics["Regime Majority Share"] = float(actual_regime_counts.max() / max(actual_regime_counts.sum(), 1))
    metrics["Regime Valid"] = float(regime_class_count == 3)
    metrics["Regime Weight"] = metrics["Regime Valid"]

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
        "regime_confusion_matrices": {
            "Persistence": persistence_cm,
            "Random Forest": rf_cm,
            "XGBoost": xgb_cm,
        },
    }


def forecast_next_steps(model, history_frame, feature_cols, steps=7, interval_low=0.0, interval_high=0.0):
    history_values = list(history_frame["groundwater_level"].values)
    history_dates = list(history_frame.index)
    if len(history_dates) >= 2:
        inferred_step = history_dates[-1] - history_dates[-2]
        step_delta = inferred_step if inferred_step > timedelta(0) else timedelta(days=1)
    else:
        step_delta = timedelta(days=1)

    rows = []
    for _ in range(steps):
        next_date = history_dates[-1] + step_delta
        series = pd.Series(history_values)

        feature_row = {
            "lag_1": series.iloc[-1],
            "lag_3": series.iloc[-3] if len(series) >= 3 else series.iloc[-1],
            "lag_7": series.iloc[-7] if len(series) >= 7 else series.iloc[0],
            "lag_14": series.iloc[-14] if len(series) >= 14 else series.iloc[0],
            "rolling_7": series.tail(7).mean(),
            "rolling_14": series.tail(14).mean(),
            "rolling_7_std": float(series.tail(7).std() or 0.0),
            "rolling_14_std": float(series.tail(14).std() or 0.0),
            "diff_1": series.iloc[-1] - series.iloc[-2] if len(series) >= 2 else 0.0,
            "diff_7": series.iloc[-1] - series.iloc[-7] if len(series) >= 7 else 0.0,
            "month": next_date.month,
            "dayofyear": next_date.dayofyear,
            "weekday": next_date.weekday(),
            "is_monsoon": int(next_date.month in [6, 7, 8, 9]),
        }

        feature_df = pd.DataFrame([feature_row])[feature_cols]
        prediction = float(model.predict(feature_df)[0])
        lower = prediction + interval_low
        upper = prediction + interval_high

        rows.append(
            {
                "date": next_date,
                "forecast": prediction,
                "lower": min(lower, upper),
                "upper": max(lower, upper),
            }
        )
        history_dates.append(next_date)
        history_values.append(prediction)

    return pd.DataFrame(rows).set_index("date")


def shap_summary(model, featured, feature_cols):
    sample = featured[feature_cols].tail(min(200, len(featured)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    return sample, shap_values


def render_kpis(featured, metrics, forecast_df):
    last_value = float(featured["groundwater_level"].iloc[-1])
    trend_window = featured["groundwater_level"].tail(min(14, len(featured)))
    trend_delta = float(trend_window.iloc[-1] - trend_window.iloc[0]) if len(trend_window) > 1 else 0.0
    row_count = len(featured)

    cols = st.columns(4)
    cols[0].metric("Latest level", f"{last_value:.2f}")
    cols[1].metric("14-day trend", f"{trend_delta:+.2f}")
    cols[2].metric("Observations used", f"{row_count}")
    cols[3].metric("Model family", "Research stack")


def render_methodology_cards(featured, feature_cols, trained):
    cleaning_steps = [
        "Column harmonization: detect groundwater timestamp/value fields across API variants.",
        "Type coercion: convert dates with `pd.to_datetime` and depths with numeric parsing.",
        "Missing-row pruning: drop rows without valid date or groundwater level.",
        "Duplicate handling: group by date and average repeated observations.",
        "Temporal ordering: sort by time before any split or feature generation.",
    ]
    feature_steps = [
        "Lag features: 1, 3, 7, and 14-step history to capture memory in the series.",
        "Rolling statistics: 7-day and 14-day mean and standard deviation for local trend/volatility.",
        "Rate-of-change terms: first and seven-step differences for depletion or recovery signals.",
        "Seasonality encodings: month, weekday, day-of-year, and monsoon indicator.",
    ]
    xai_steps = [
        "SHAP TreeExplainer on XGBoost to quantify per-feature contribution.",
        "Bar summary plot for global feature ranking and interpretation.",
        "Top SHAP drivers surfaced as text and a heatmap so non-technical users can follow the logic.",
    ]

    left, middle, right = st.columns(3)
    with left:
        st.markdown("### Data Cleaning")
        for step in cleaning_steps:
            st.markdown(f"- {step}")
    with middle:
        st.markdown("### Feature Extraction")
        for step in feature_steps:
            st.markdown(f"- {step}")
    with right:
        st.markdown("### XAI")
        for step in xai_steps:
            st.markdown(f"- {step}")

    summary = pd.DataFrame(
        {
            "Metric": [
                "Cleaned rows",
                "Engineered features",
                "Train rows",
                "Test rows",
                "Anomaly rate (%)",
            ],
            "Value": [
                len(featured),
                len(feature_cols),
                len(trained["train_df"]),
                len(trained["test_df"]),
                round(float(trained["featured"]["anomaly_flag"].mean() * 100), 2),
            ],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_model_explainer_dialogs():
    with st.expander("Why anomaly detection?", expanded=False):
        st.markdown(
            """
            Anomaly detection flags readings that do not fit the normal short-term pattern. For groundwater, this helps
            separate unusual sensor events, abrupt drops or rises, and noisy records from the regular seasonal movement.
            The dashboard uses Isolation Forest because it can learn what looks normal from the full series without needing
            manual labels for every abnormal case.
            """
        )
        st.caption("In simple terms: it highlights readings that deserve a second look before decisions are made.")

    with st.expander("Why forecasting?", expanded=False):
        st.markdown(
            """
            Forecasting gives a short-term view of where the groundwater level is heading. That is useful for early warning,
            comparison against the latest observation, and simple planning. The model uses recent history, rolling averages,
            and seasonality so the prediction is based on actual temporal context rather than a single reading.
            """
        )
        st.caption("In simple terms: it answers what is likely to happen next, not just what happened today.")


def describe_groundwater_status(featured, forecast_df):
    latest_level = float(featured["groundwater_level"].iloc[-1])
    recent_window = featured["groundwater_level"].tail(min(14, len(featured)))
    recent_change = float(recent_window.iloc[-1] - recent_window.iloc[0]) if len(recent_window) > 1 else 0.0
    next_forecast = float(forecast_df["forecast"].iloc[0]) if forecast_df is not None and not forecast_df.empty else np.nan
    anomaly_rate = float(featured["anomaly_flag"].mean() * 100)

    if recent_change > 0.5:
        trend_text = "The recent pattern is moving upward, which usually means groundwater depth is increasing and conditions are becoming less favorable."
    elif recent_change < -0.5:
        trend_text = "The recent pattern is moving downward, which usually means groundwater depth is falling and conditions are improving."
    else:
        trend_text = "The recent pattern is mostly stable, so groundwater conditions are not changing sharply right now."

    if np.isfinite(next_forecast):
        if next_forecast > latest_level:
            forecast_text = "The short-term forecast suggests the level may rise further, so the next few days should be watched carefully."
        elif next_forecast < latest_level:
            forecast_text = "The short-term forecast suggests a small improvement compared with the latest observed level."
        else:
            forecast_text = "The short-term forecast is close to the current level, which indicates stability."
    else:
        forecast_text = "A short-term forecast is not available because the model could not generate enough future steps."

    if anomaly_rate >= 10:
        anomaly_text = "The anomaly rate is high, which means the series contains several unusual readings that should be checked before drawing a strong conclusion."
    elif anomaly_rate >= 5:
        anomaly_text = "The anomaly rate is moderate, so the data looks mostly normal but still has a few unusual points."
    else:
        anomaly_text = "The anomaly rate is low, so the recent data looks fairly consistent."

    status_score = recent_change + (next_forecast - latest_level if np.isfinite(next_forecast) else 0.0)
    if status_score > 1.0:
        status_label = "Stress increasing"
    elif status_score < -1.0:
        status_label = "Condition improving"
    else:
        status_label = "Broadly stable"

    return {
        "status_label": status_label,
        "latest_level": latest_level,
        "recent_change": recent_change,
        "next_forecast": next_forecast,
        "anomaly_rate": anomaly_rate,
        "trend_text": trend_text,
        "forecast_text": forecast_text,
        "anomaly_text": anomaly_text,
    }


def describe_xai_meaning(feature_cols, shap_values):
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    top_features = importance.head(5)

    if len(top_features) == 0:
        summary_text = "The explanation layer could not identify clear drivers for this sample."
    else:
        readable_names = [_format_driver_name(feature_name) for feature_name in top_features.index]

        summary_text = (
            "The model is mainly paying attention to "
            + ", ".join(readable_names[:3])
            + ". This means the forecast is driven more by recent groundwater behavior and seasonality than by a single isolated reading."
        )

    return {
        "top_features": top_features,
        "summary_text": summary_text,
        "plain_language": (
            "Positive SHAP impact pushes the prediction toward deeper groundwater, while negative SHAP impact pulls it toward a shallower level. "
            "In simple words, SHAP shows which past water-level patterns and seasonal signals are making the model predict a rise or fall."
        ),
    }


def render_map_panel(location_context, featured):
    st.subheader("Geographical view")
    if location_context is None:
        st.info("No coordinates were found in the incoming data. The map panel falls back to a district-level location when possible.")
        return

    st.caption(f"Map source: {location_context['source']}")
    map_fig = build_location_map(location_context, featured)
    if map_fig is None:
        st.info("A map could not be built from the current data.")
        return

    _plotly_chart(map_fig, key="home_map")
    st.markdown(
        f"""
        <div style="padding:0.9rem 1rem;border-radius:14px;background:#f8fbfd;border:1px solid #dbe7ef;">
            <strong>Selected area:</strong> {location_context['map_label']}<br/>
            <strong>Latest groundwater level:</strong> {float(featured['groundwater_level'].iloc[-1]):.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pattern_panel(trained, feature_cols):
    st.subheader("Patterns and validation heatmaps")
    corr_frame = correlation_heatmap_frame(trained["featured"], feature_cols)
    corr_fig = build_correlation_heatmap(corr_frame)
    _plotly_chart(corr_fig, key="research_correlation")
    st.caption("This correlation matrix shows which engineered signals move together and which ones behave differently.")

    st.markdown("### Model scorecard")
    metric_fig = go.Figure()
    for metric_name in ["RMSE", "MAE", "MAPE"]:
        metric_fig.add_trace(
            go.Bar(
                x=trained["metrics"].index,
                y=trained["metrics"][metric_name],
                name=metric_name,
            )
        )
    metric_fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    _plotly_chart(metric_fig, key="research_metrics_bars")

    render_confusion_matrix_panel(trained)


def render_benchmark_panel(trained):
    benchmark = build_benchmark_summary(exclude_latest=True)
    st.subheader("Cumulative research benchmark")

    if benchmark is None:
        st.info("No prior benchmark history exists yet. This panel is reserved for historical comparison across earlier runs and stays separate from the current dataset scorecard.")
        return

    st.caption(f"Pooled over {len(benchmark['runs'])} prior runs only. These are historical comparison values, not the current dataset scorecard.")
    st.dataframe(benchmark["summary"].style.format("{:.4f}"), use_container_width=True)

    pooled_fig = go.Figure()
    for metric_name in ["RMSE", "MAE", "MAPE"]:
        pooled_fig.add_trace(
            go.Bar(
                x=benchmark["summary"].index,
                y=benchmark["summary"][metric_name],
                name=metric_name,
            )
        )
    pooled_fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    _plotly_chart(pooled_fig, key="research_pooled_metrics")

    st.markdown("### Cumulative confusion matrix")
    heat_fig = go.Figure(
        data=go.Heatmap(
            z=benchmark["xgb_cm"].values,
            x=benchmark["xgb_cm"].columns.tolist(),
            y=benchmark["xgb_cm"].index.tolist(),
            colorscale="Blues",
            showscale=True,
            text=benchmark["xgb_cm"].values,
            texttemplate="%{text}",
        )
    )
    heat_fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    _plotly_chart(heat_fig, key="research_pooled_confusion_matrix")


def render_research_protocol(trained, feature_cols, shap_values):
    st.subheader("Research protocol")
    protocol_cols = st.columns(3)
    protocol_cols[0].metric("Train/Test split", "80/20")
    protocol_cols[1].metric("Target", "Groundwater level")
    protocol_cols[2].metric("Primary model", "XGBoost")

    st.markdown(
        """
        The current-run metrics are computed only on the selected dataset and use a chronological hold-out test period. This is standard forecasting evaluation: train on earlier observations, test on later unseen observations, and compare predictions against the actual test values. The numbers can differ across states or districts because each area has a different groundwater regime, sample length, seasonal structure, and variance. The pooled benchmark below is only a historical comparison across saved runs.
        """
    )
    st.markdown("- RMSE measures absolute forecast error in the original groundwater units.")
    st.markdown("- MAE measures the average absolute deviation and is easier to read than RMSE.")
    st.markdown("- MAPE reports percentage error, which helps compare across datasets with different scales.")
    st.markdown("- R2, NSE, and KGE are standard hydrology/forecasting skill measures for fit and efficiency.")
    st.markdown("- NRMSE normalizes the RMSE by the observed range so results can be compared across districts.")
    st.markdown("- PBIAS and Bias show whether the model systematically overpredicts or underpredicts.")
    st.markdown("- Pearson r shows whether the predicted and actual series move together.")
    st.markdown("- Regime accuracy and confusion matrices convert the continuous series into Low / Moderate / High classes using train-set quantiles.")
    st.markdown("- SHAP explains which engineered lags, rolling values, and seasonal variables changed the model output.")
    st.info(trained.get("regime_eval_note", ""))

    st.markdown("### Current dataset scorecard")
    st.dataframe(trained["metrics"].style.format("{:.4f}"), use_container_width=True)

    st.caption("Current-run metrics are the main research result. Historical pooled metrics are shown separately so the two do not get mixed together. This prevents a district-specific score from being mistaken for a cross-district benchmark.")


def render_forecast_and_anomaly_panel(trained, forecast_df):
    st.subheader("Forecast and anomaly view")
    anomaly_fig = build_anomaly_chart(trained["featured"])
    _plotly_chart(anomaly_fig, key="home_anomaly_chart")

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=trained["featured"].index,
            y=trained["featured"]["groundwater_level"],
            name="Historical",
            line=dict(color="#2c7fb8", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast"],
            name="Forecast",
            line=dict(color="#f28e2b", width=3, dash="dash"),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
            y=forecast_df["upper"].tolist() + forecast_df["lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(242, 142, 43, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence band",
            hoverinfo="skip",
        )
    )
    forecast_fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    _plotly_chart(forecast_fig, key="home_forecast_chart")
    st.dataframe(forecast_df, use_container_width=True)


def render_confusion_matrix_panel(trained):
    st.subheader("Regime-based confusion matrix")
    st.caption("Because the target is continuous, the dashboard converts groundwater levels into Low / Moderate / High regimes using train-set quantiles, then compares predicted regimes against actual regimes.")

    if float(trained["metrics"].iloc[0]["Regime Valid"]) < 1.0:
        st.warning(
            "Regime accuracy is displayed for transparency, but this split does not contain all three regime classes. Treat the regime number as informational only and rely primarily on RMSE, MAE, NSE, KGE, and MAPE for comparison."
        )

    matrix_choice = st.selectbox("Choose model", ["Persistence", "Random Forest", "XGBoost"], index=2, key="regime_model_choice")
    matrix_df = trained["regime_confusion_matrices"][matrix_choice]

    heat_fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=["Pred Low", "Pred Moderate", "Pred High"],
            y=["Actual Low", "Actual Moderate", "Actual High"],
            colorscale="Blues",
            showscale=True,
            text=matrix_df.values,
            texttemplate="%{text}",
        )
    )
    heat_fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    _plotly_chart(heat_fig, key="research_regime_confusion_matrix")

    st.dataframe(matrix_df, use_container_width=True)

    regime_accuracy = trained["metrics"]["Regime Acc"].rename("Regime Acc")
    st.write("Regime accuracy by model")
    st.dataframe(regime_accuracy.to_frame(), use_container_width=True)

    coverage = trained["metrics"].loc[:, ["Regime Coverage", "Regime Majority Share"]].head(1)
    st.write("Regime coverage diagnostics")
    st.dataframe(coverage.style.format("{:.3f}"), use_container_width=True)
    st.caption(trained.get("regime_eval_note", ""))


def render_plain_language_panels(trained, forecast_df, feature_cols, shap_values):
    groundwater_story = describe_groundwater_status(trained["featured"], forecast_df)
    xai_story = describe_xai_meaning(feature_cols, shap_values)
    next_forecast_display = f"{groundwater_story['next_forecast']:.2f}" if np.isfinite(groundwater_story["next_forecast"]) else "N/A"

    st.subheader("Plain-language groundwater status")
    st.markdown(
        f"""
        <div style="padding:1rem 1.1rem;border-radius:16px;background:#ffffff;border:1px solid #dfe8ec;color:#10313e;">
            <p style="margin:0 0 0.35rem 0;font-weight:700;font-size:1.05rem;">Current status: {groundwater_story['status_label']}</p>
            <p style="margin:0 0 0.35rem 0;">Latest observed level: {groundwater_story['latest_level']:.2f}</p>
            <p style="margin:0 0 0.35rem 0;">Recent change over the last 14 observations: {groundwater_story['recent_change']:+.2f}</p>
            <p style="margin:0 0 0.35rem 0;">Next forecasted level: {next_forecast_display}</p>
            <p style="margin:0 0 0.35rem 0;">Anomaly rate: {groundwater_story['anomaly_rate']:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(groundwater_story["trend_text"])
    st.info(groundwater_story["forecast_text"])
    st.info(groundwater_story["anomaly_text"])

    st.subheader("Plain-language XAI meaning")
    st.markdown(
        f"""
        <div style="padding:1rem 1.1rem;border-radius:16px;background:#ffffff;border:1px solid #dfe8ec;color:#10313e;">
            <p style="margin:0 0 0.35rem 0;font-weight:700;font-size:1.05rem;">What SHAP is saying</p>
            <p style="margin:0;">{xai_story['summary_text']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(xai_story["plain_language"])
    st.dataframe(xai_story["top_features"].rename("mean_abs_shap").to_frame(), use_container_width=True)


def render_transparency_explainers(trained, forecast_df, feature_cols, shap_values):
    with st.expander("How to read this dashboard", expanded=True):
        st.markdown(
            """
            This dashboard follows a research-style flow: raw groundwater values are cleaned first, then transformed into lag and rolling features, then evaluated with a train/test split, and finally interpreted with SHAP. The figures are chosen so a non-technical user can follow the logic without reading code.
            """
        )
        st.markdown("- Correlation matrix: shows whether engineered inputs move together or overlap.")
        st.markdown("- Confusion matrix: converts continuous groundwater into Low / Moderate / High regime classes so the model can be checked in a policy-friendly way.")
        st.markdown("- SHAP bar plot: ranks the strongest drivers of the forecast.")
        st.markdown("- SHAP heatmap: shows how those drivers change across recent dates.")
        st.markdown("- Forecast plot: shows short-term projection and uncertainty band.")
        st.markdown("- Anomaly chart: highlights unusual readings that should be reviewed before using the results.")

    with st.expander("Why these models are used", expanded=False):
        st.markdown(
            """
            Random Forest and XGBoost are included because they handle nonlinear time-series interactions well. Isolation Forest is used for anomaly detection because it can identify unusual patterns without requiring manual labels for every abnormal reading. The baseline persistence model provides a simple reference point so the research comparison remains defensible.
            """
        )

    with st.expander("What SHAP means here", expanded=False):
        st.markdown(
            """
            SHAP is a transparency layer. A positive SHAP value pushes the model toward a deeper groundwater prediction; a negative value pulls it toward a shallower prediction. The goal is not just to predict, but to explain which recent readings and seasonal signals changed the output.
            """
        )

    with st.expander("How anomaly and forecasting should be read", expanded=False):
        st.markdown(
            """
            Anomaly detection marks readings that are different from the learned normal pattern. Forecasting gives the likely next movement based on the recent sequence. They answer different questions: anomaly detection asks whether a point is unusual, while forecasting asks where the series is likely to go next.
            """
        )


def main():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #f4faf9;
            color: #10313e;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e7eef5;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 6px 20px rgba(16, 24, 40, 0.04);
        }
        [data-testid="stDataFrame"] {
            background: #ffffff;
        }
        section[data-testid="stSidebar"] {
            background: #f7fbfb;
        }
        .hero {
            padding: 1.1rem 1.3rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 55%, #e3f1f2 100%);
            color: #10313e;
            margin-bottom: 1rem;
            border: 1px solid #d7e7ea;
            box-shadow: 0 12px 28px rgba(16, 49, 62, 0.07);
        }
        .subtle { opacity: 0.78; }
        .section-card {
            background: #fbfefe;
            border: 1px solid #e3eef1;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(16, 49, 62, 0.03);
        }
        .stButton button,
        .stSelectbox [data-baseweb="select"],
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSlider {
            background: #ffffff !important;
            color: #10313e !important;
        }
        h1, h2, h3, h4, p, label, .stMarkdown, .stCaption {
            color: #10313e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">JalDarpan</h1>
            <p class="subtle" style="margin:0.35rem 0 0 0;">Simple groundwater insights for day-to-day use, with a separate research view for detailed evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inputs")
        backend_url = st.text_input("Backend URL", DEFAULT_BACKEND_URL)
        state = st.text_input("State", "Odisha")
        district = st.text_input("District", "Baleshwar")
        agency = st.text_input("Agency", "CGWB")
        startdate = st.date_input("Start date", datetime.now() - timedelta(days=365))
        enddate = st.date_input("End date", datetime.now())
        size = st.number_input("Max records", min_value=50, max_value=2000, value=500, step=50)
        forecast_horizon = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7)
        run_button = st.button("Run analysis", type="primary")

    if run_button:
        with st.spinner("Fetching and preparing groundwater data..."):
            try:
                raw_df = fetch_groundwater_data(
                    backend_url,
                    state,
                    district,
                    agency,
                    startdate.strftime("%Y-%m-%d"),
                    enddate.strftime("%Y-%m-%d"),
                    int(size),
                )
            except Exception as exc:
                st.warning(f"Live fetch failed: {exc}")
                raw_df = load_cached_data()

        if raw_df is None or raw_df.empty:
            st.error("No groundwater data available from live API or local cache.")
            return

        featured = normalize_groundwater_frame(raw_df)
        if featured is None or featured.empty:
            st.error("Could not normalize the incoming data. Check the API field names.")
            return

        featured, feature_cols = build_features(featured)
        if featured is None or featured.empty:
            st.error("Not enough dated data after feature engineering. Increase the date range.")
            return

        trained = train_models(featured, feature_cols)
        if trained is None:
            st.error("Not enough samples to train the research models.")
            return

        location_context = build_location_context(raw_df, state, district)

        forecast_df = forecast_next_steps(
            trained["xgb"],
            featured[["groundwater_level"]],
            feature_cols,
            steps=int(forecast_horizon),
            interval_low=trained["interval_low"],
            interval_high=trained["interval_high"],
        )

        record_benchmark_run(
            {
                "state": state,
                "district": district,
                "agency": agency,
                "rows": len(trained["featured"]),
            },
            trained["metrics"],
            trained,
        )

        render_kpis(trained["featured"], trained["metrics"], forecast_df)

        home_tab, research_tab = st.tabs(["Home", "Research"])

        with home_tab:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Live groundwater view")
            st.write(
                "A simple snapshot for non-technical users: the selected area, the latest groundwater movement, unusual readings, and a short-term forecast with plain-language interpretation."
            )
            st.dataframe(trained["featured"].tail(12), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.divider()
            render_map_panel(location_context, trained["featured"])

            st.divider()
            render_forecast_and_anomaly_panel(trained, forecast_df)

            st.divider()
            st.subheader("Simple explanation")
            render_model_explainer_dialogs()
            sample, shap_values = shap_summary(trained["xgb"], trained["featured"], feature_cols)
            top_features = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
            st.write(
                "The model is mainly using recent groundwater history and seasonal patterns to make its next-step estimate. The strongest drivers are: "
                + ", ".join(top_features.head(5).index.tolist())
            )
            st.dataframe(top_features.rename("mean_abs_shap").to_frame(), use_container_width=True)
            st.info(
                "SHAP shows which recent signals are pushing the forecast up or down. Anomaly detection flags unusual points so users can review them before trusting the trend."
            )
            render_plain_language_panels(trained, forecast_df, feature_cols, shap_values)

        with research_tab:
            st.subheader("Research metrics and graphics")
            st.caption("This tab keeps the research-style evaluation separate from the normal user view.")
            render_research_protocol(trained, feature_cols, shap_values)
            st.divider()
            render_pattern_panel(trained, feature_cols)
            st.divider()
            render_benchmark_panel(trained)

            st.divider()
            st.subheader("Detailed explainability")
            sample, shap_values = shap_summary(trained["xgb"], trained["featured"], feature_cols)
            bar_fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            st.pyplot(bar_fig, clear_figure=True)

            dot_fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample, show=False)
            st.pyplot(dot_fig, clear_figure=True)

            shap_heatmap = build_shap_heatmap(sample, shap_values, feature_cols)
            if shap_heatmap is not None:
                _plotly_chart(shap_heatmap, key="research_shap_heatmap")

            st.write("Top drivers: " + ", ".join(pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False).head(5).index.tolist()))
            st.info(
                "The research view follows the paper-style structure: feature ranking, SHAP spread, temporal contribution heatmap, correlation matrix, regime confusion matrix, and pooled performance summary."
            )

            st.divider()
            render_transparency_explainers(trained, forecast_df, feature_cols, shap_values)

            st.divider()
            render_methodology_cards(trained["featured"], feature_cols, trained)
            st.markdown("### Research data")
            st.dataframe(raw_df, use_container_width=True)
            st.download_button(
                "Download cleaned features as CSV",
                data=trained["featured"].reset_index().to_csv(index=False).encode("utf-8"),
                file_name="jaldarpan_features.csv",
                mime="text/csv",
            )

        st.success("Dashboard ready: map, heatmaps, explainability, and research validation are available in a simplified layout.")

    else:
        st.info("Choose your parameters in the sidebar and run the analysis.")


if __name__ == "__main__":
    main()