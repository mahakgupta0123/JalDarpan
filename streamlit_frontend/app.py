from datetime import datetime, timedelta
import os
import sys

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

# Import benchmark utilities from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from benchmark_utils import load_benchmark_results, build_pooled_summaries, get_regime_validity_note

st.set_page_config(page_title="JalDarpan", page_icon="🌊", layout="wide")

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
BENCHMARK_CSV = os.environ.get("BENCHMARK_CSV", os.path.join(parent_dir, "benchmark_results.csv"))
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


def safe_st_dataframe(frame, **kwargs):
    if frame is None:
        st.write("No data available.")
        return
    try:
        st.dataframe(frame, **kwargs)
        return
    except Exception:
        sanitized = frame.copy()
        for col in sanitized.columns:
            if sanitized[col].dtype == object:
                sanitized[col] = sanitized[col].astype(str)
        try:
            st.dataframe(sanitized, **kwargs)
            return
        except Exception:
            st.write(sanitized)


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


def _fetch_backend_list(backend_url, endpoint, path_value=None):
    if not backend_url:
        return []
    url = f"{backend_url.rstrip('/')}/{endpoint.lstrip('/')}"
    if path_value:
        url = f"{url}/{requests.utils.requote_uri(path_value)}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            return []
        return payload.get(endpoint.strip('/'), []) or []
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_state_choices(backend_url):
    return _fetch_backend_list(backend_url, "states")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_district_choices(backend_url, state):
    if not state:
        return []
    return _fetch_backend_list(backend_url, "districts", path_value=state)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_agency_choices(backend_url):
    return _fetch_backend_list(backend_url, "agencies")


def load_benchmark_runs():
    """Load benchmark runs from CSV. Falls back to old cache if new CSV not available."""
    results_df = load_benchmark_results(BENCHMARK_CSV)
    if results_df is not None and not results_df.empty:
        return results_df
    
    # Fallback to old cache file for backward compatibility
    if os.path.exists(BENCHMARK_CACHE):
        return pd.read_csv(BENCHMARK_CACHE)
    
    return pd.DataFrame()


def record_benchmark_run(meta, metrics, trained):
    """
    Record a single benchmark run to the local cache CSV.
    This is still used by the live dashboard for accumulating manual runs.
    """
    row = {
        "state": meta.get("state", ""),
        "district": meta.get("district", ""),
        "agency": meta.get("agency", ""),
        "rows": int(meta.get("rows", 0)),
        "split_ratio": trained.get("split_ratio", 0.80),
        "regime_valid": trained.get("regime_valid", False),
        "regime_class_count": trained.get("regime_class_count", 0),
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
    """
    Build a benchmark summary from loaded runs.
    If BENCHMARK_CSV exists and is populated, use that as primary source.
    Otherwise fall back to local cache.
    """
    runs = load_benchmark_runs()
    if runs.empty:
        return None

    if exclude_latest and len(runs) > 1:
        runs = runs.iloc[:-1].copy()

    # Only include successful runs
    if "status" in runs.columns:
        runs = runs[runs["status"] == "SUCCESS"].copy()
    
    if runs.empty:
        return None

    pooled = build_pooled_summaries(runs)
    if pooled is None:
        return None

    return {
        "runs": runs,
        "summary": pooled["summary"],
        "xgb_cm": pooled.get("xgb_cm", pd.DataFrame()),
    }


def _safe_normalize(series):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index)
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v == min_v:
        return pd.Series(0.5, index=series.index)
    return (series - min_v) / (max_v - min_v)


def build_current_dataset_summary(state, district, featured, trained, forecast_df):
    latest = float(featured["groundwater_level"].iloc[-1])
    mean_level = float(featured["groundwater_level"].mean())
    std_level = float(featured["groundwater_level"].std())
    min_level = float(featured["groundwater_level"].min())
    max_level = float(featured["groundwater_level"].max())
    normalized_latest = (latest - min_level) / (max_level - min_level) if max_level != min_level else 0.5
    first_date = featured.index.min()
    last_date = featured.index.max()
    forecast_next = float(forecast_df["forecast"].iloc[0]) if forecast_df is not None and not forecast_df.empty else np.nan
    regime_label = None
    if trained.get("regime_bins") is not None:
        regime_label = pd.cut(pd.Series([latest]), bins=trained["regime_bins"], labels=trained["regime_labels"], include_lowest=True)[0]

    summary = pd.DataFrame(
        {
            "Metric": [
                "Selected state",
                "Selected district",
                "Observations used",
                "Date range",
                "Latest groundwater level",
                "Normalized latest level",
                "Mean level",
                "Level standard deviation",
                "Low/high range",
                "Next forecast",
                "Forecast horizon",
                "Current regime",
                "Anomaly rate",
            ],
            "Value": [
                state,
                district,
                len(featured),
                f"{first_date.date()} → {last_date.date()} ({(last_date - first_date).days} days)",
                f"{latest:.3f}",
                f"{normalized_latest:.2f}",
                f"{mean_level:.3f}",
                f"{std_level:.3f}",
                f"{min_level:.3f} → {max_level:.3f}",
                f"{forecast_next:.3f}" if np.isfinite(forecast_next) else "N/A",
                "short-term",
                str(regime_label or "unclassified"),
                f"{float(trained['featured']['anomaly_flag'].mean() * 100):.1f}%",
            ],
        }
    )
    summary["Value"] = summary["Value"].astype(str)
    return summary


def build_benchmark_aggregates(runs):
    if runs is None or runs.empty:
        return {}

    numeric = runs.copy()
    for col in [
        "XGBoost__MAPE",
        "XGBoost__RMSE",
        "XGBoost__NRMSE",
        "XGBoost__Regime Valid",
        "XGBoost__Regime Coverage",
        "XGBoost__Regime Majority Share",
    ]:
        if col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    if "state" in numeric.columns:
        state_summary = numeric.groupby("state").agg(
            districts=("district", "nunique"),
            avg_mape=("XGBoost__MAPE", "mean") if "XGBoost__MAPE" in numeric.columns else ("rows", "count"),
            avg_rmse=("XGBoost__RMSE", "mean") if "XGBoost__RMSE" in numeric.columns else ("rows", "count"),
            avg_nrmse=("XGBoost__NRMSE", "mean") if "XGBoost__NRMSE" in numeric.columns else ("rows", "count"),
            regime_valid_pct=("XGBoost__Regime Valid", "mean") if "XGBoost__Regime Valid" in numeric.columns else ("rows", "count"),
        ).reset_index()
        if "avg_mape" in state_summary.columns:
            state_summary = state_summary.sort_values("avg_mape")
    else:
        state_summary = pd.DataFrame()

    national_summary = build_benchmark_summary().get("summary") if build_benchmark_summary() is not None else pd.DataFrame()

    return {
        "state_summary": state_summary,
        "national_summary": national_summary,
    }


def build_alert_rankings(runs):
    if runs is None or runs.empty:
        return {}

    ranking = runs.copy()
    for col in [
        "XGBoost__MAPE",
        "XGBoost__NRMSE",
        "XGBoost__Regime Valid",
        "XGBoost__Regime Majority Share",
    ]:
        if col in ranking.columns:
            ranking[col] = pd.to_numeric(ranking[col], errors="coerce")

    ranking["risk_score"] = (
        ranking.get("XGBoost__MAPE", 0).fillna(0) * 0.65
        + ranking.get("XGBoost__NRMSE", 0).fillna(0) * 0.35
    )

    top_alerts = ranking.sort_values("risk_score", ascending=False).head(10)
    top_adequate = ranking.sort_values("XGBoost__MAPE", ascending=True).head(10)
    top_high_confidence = ranking[ranking.get("XGBoost__Regime Valid", 0) > 0].sort_values(
        "XGBoost__Regime Majority Share", ascending=False
    ).head(10)

    return {
        "alerts": top_alerts[["state", "district", "risk_score", "XGBoost__MAPE", "XGBoost__NRMSE"]].copy() if not top_alerts.empty else pd.DataFrame(),
        "adequate": top_adequate[["state", "district", "XGBoost__MAPE", "XGBoost__NRMSE"]].copy() if not top_adequate.empty else pd.DataFrame(),
        "high_confidence": top_high_confidence[["state", "district", "XGBoost__Regime Majority Share", "XGBoost__Regime Coverage"]].copy() if not top_high_confidence.empty else pd.DataFrame(),
    }


def render_dataset_analysis_panel(state, district, featured, trained, forecast_df, benchmark_aggregates):
    st.subheader("Dataset analysis")
    current_summary = build_current_dataset_summary(state, district, featured, trained, forecast_df)
    safe_st_dataframe(current_summary, use_container_width=True)

    if benchmark_aggregates.get("state_summary") is not None and not benchmark_aggregates["state_summary"].empty:
        st.markdown("### State-level benchmark summary")
        st.dataframe(
            benchmark_aggregates["state_summary"].head(12).style.format(
                {
                    "avg_mape": "{:.2f}",
                    "avg_rmse": "{:.3f}",
                    "avg_nrmse": "{:.3f}",
                    "regime_valid_pct": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    if benchmark_aggregates.get("national_summary") is not None and not benchmark_aggregates["national_summary"].empty:
        st.markdown("### National pooled benchmark")
        st.dataframe(benchmark_aggregates["national_summary"].style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Feature engineering overview")
    st.write(
        f"- Generated {len(trained['feature_cols'])} engineered features from the cleaned time series."
    )
    st.write("- Normalized the most recent groundwater value relative to the selected district range.")
    st.write("- Used rolling means, rolling standard deviations, lag values, differences, and seasonal encodings.")


def render_alert_tables(runs):
    st.subheader("Risk and water-level alert board")
    rankings = build_alert_rankings(runs)
    if not rankings:
        st.info("No benchmark alert rankings are available yet.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Top 10 alarmed districts")
        if not rankings["alerts"].empty:
            st.dataframe(
                rankings["alerts"].rename(
                    columns={
                        "risk_score": "Risk score",
                        "XGBoost__MAPE": "MAPE",
                        "XGBoost__NRMSE": "NRMSE",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.write("No high-risk districts found.")
    with cols[1]:
        st.markdown("#### Top 10 adequate districts")
        if not rankings["adequate"].empty:
            st.dataframe(
                rankings["adequate"].rename(
                    columns={
                        "XGBoost__MAPE": "MAPE",
                        "XGBoost__NRMSE": "NRMSE",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.write("No adequate districts can be ranked right now.")
    with cols[2]:
        st.markdown("#### Top 10 high-confidence districts")
        if not rankings["high_confidence"].empty:
            st.dataframe(
                rankings["high_confidence"].rename(
                    columns={
                        "XGBoost__Regime Majority Share": "Regime majority",
                        "XGBoost__Regime Coverage": "Regime coverage",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.write("No high-confidence districts with valid regime coverage are available.")


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
    gwl_min = float(np.nanmin(y_true)) if len(y_true) > 0 else np.nan
    gwl_max = float(np.nanmax(y_true)) if len(y_true) > 0 else np.nan
    spread = (gwl_max - gwl_min) if (not np.isnan(gwl_max) and not np.isnan(gwl_min)) else 0.0
    mean_true = float(np.mean(y_true)) or 1.0

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / np.maximum(np.abs(y_true), 1e-6))) * 100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    nse = float(1.0 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) > 1 and not np.isclose(np.sum((y_true - np.mean(y_true)) ** 2), 0.0) else np.nan
    nrmse = float(rmse / spread) if spread > 0 else np.nan
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
        "gwl_min": gwl_min,
        "gwl_max": gwl_max,
        "gwl_range": spread,
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
        go.Scattermapbox(
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
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=7),
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


def build_daily_groundwater_heatmap(featured):
    if featured is None or featured.empty:
        return None
    if "groundwater_level" not in featured.columns:
        return None
    levels = pd.to_numeric(featured["groundwater_level"], errors="coerce")
    if levels.isna().all():
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=[levels.fillna(0.0).tolist()],
            x=featured.index,
            y=["Groundwater level"],
            colorscale="Viridis",
            colorbar=dict(title="Level"),
            hovertemplate="Date: %{x}<br>Level: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#10313e"),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def _build_placeholder_heatmap(title, message):
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=f"<b>{title}</b><br>{message}",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14, color="#10313e"),
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def build_correlation_heatmap(corr_frame):
    if corr_frame is None or corr_frame.empty:
        return _build_placeholder_heatmap(
            "Correlation heatmap unavailable",
            "No numeric feature correlations could be computed from the selected data.",
        )

    corr_frame = corr_frame.copy()
    corr_frame = corr_frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr_frame.empty:
        return _build_placeholder_heatmap(
            "Correlation heatmap unavailable",
            "All correlation data is empty or invalid.",
        )

    try:
        z_values = corr_frame.astype(float).fillna(0.0).values
    except Exception:
        try:
            z_values = pd.to_numeric(corr_frame.values, errors="coerce")
            z_values = np.nan_to_num(z_values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return _build_placeholder_heatmap(
                "Correlation heatmap unavailable",
                "Could not convert correlation values to numbers.",
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=corr_frame.columns.tolist(),
            y=corr_frame.index.tolist(),
            colorscale="YlGnBu",
            zmid=0,
            text=np.round(z_values, 3),
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
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
        "split_ratio": 0.80,  # Dashboard uses fixed 80/20; sweep script uses ladder
        "regime_valid": regime_class_count == 3,
        "regime_class_count": regime_class_count,
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
        "Top SHAP drivers surfaced as text for easy interpretation.",
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
    summary["Value"] = summary["Value"].astype(str)
    safe_st_dataframe(summary, use_container_width=True, hide_index=True)


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
    st.subheader("Patterns and validation")
    st.markdown("This section summarizes the research validation metrics and model performance, including the feature correlation heatmap for the selected dataset.")

    corr_frame = correlation_heatmap_frame(trained["featured"], feature_cols)
    corr_fig = build_correlation_heatmap(corr_frame)
    _plotly_chart(corr_fig, key="research_correlation_heatmap")
    st.caption("The correlation heatmap shows how the engineered features move together and which signals are most closely related to groundwater level.")

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

    if benchmark is None or benchmark["summary"].empty:
        st.info("No benchmark runs available yet. Run `python run_benchmark_sweep.py` from the project root to populate the national benchmark, or navigate to Settings > Run analysis to record manual runs here.")
        return

    runs = benchmark["runs"]
    valid_run_count = (runs.get("regime_valid", False).sum() if "regime_valid" in runs.columns else 0)
    total_run_count = len(runs)
    
    validity_note = get_regime_validity_note(valid_run_count, total_run_count)
    st.caption(f"Pooled over {total_run_count} prior runs. {validity_note}")
    
    # --- Per-District Table ---
    st.markdown("### Per-district results")
    
    # Select display columns
    display_cols = ["state", "district", "split_ratio", "regime_valid", "train_rows", "test_rows", "anomaly_rate_pct"]
    
    # Add model summary metrics (RMSE, MAE, MAPE from XGBoost)
    for metric in ["RMSE", "MAE", "MAPE"]:
        col_name = f"XGBoost__{metric}"
        if col_name in runs.columns:
            display_cols.append(col_name)
    
    display_df = runs[display_cols].copy() if all(col in runs.columns for col in display_cols) else runs
    
    st.dataframe(
        display_df.style.format({
            "split_ratio": "{:.0%}",
            "anomaly_rate_pct": "{:.1f}%",
            "XGBoost__RMSE": "{:.3f}",
            "XGBoost__MAE": "{:.3f}",
            "XGBoost__MAPE": "{:.1f}",
        }),
        use_container_width=True,
        height=400,
    )
    st.caption("split_ratio: train/test ratio used (80/20 preferred, falling back to 75/25 or 70/30 if 80/20 didn't yield 3 regime classes). regime_valid: whether all 3 regime classes appeared in test set. anomaly_rate_pct: % of observations flagged as anomalous by Isolation Forest.")
    
    # --- Aggregated Metrics ---
    st.markdown("### Pooled national summary")
    st.dataframe(benchmark["summary"].style.format("{:.4f}"), use_container_width=True)

    # --- Metrics Bars ---
    pooled_fig = go.Figure()
    for metric_name in ["RMSE", "MAE", "MAPE"]:
        if metric_name in benchmark["summary"].columns:
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

    st.markdown("### Cumulative confusion matrix (XGBoost)")
    if not benchmark["xgb_cm"].empty:
        st.dataframe(benchmark["xgb_cm"].astype(str), use_container_width=True)
    else:
        st.info("Cumulative confusion matrix data is unavailable.")
    st.info(validity_note)


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

    water_heatmap = build_daily_groundwater_heatmap(trained["featured"])
    if water_heatmap is not None:
        st.markdown("### Daily groundwater heatmap")
        _plotly_chart(water_heatmap, key="home_water_level_heatmap")
    else:
        st.info("Daily groundwater heatmap is not available for this dataset.")

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
    safe_st_dataframe(forecast_df, use_container_width=True)


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


def render_national_aggregation_dashboard():
    """Render the national-level aggregation metrics dashboard."""
    import traceback
    
    try:
        # Try to find the aggregation_outputs directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "aggregation_outputs")
        
        st.info(f"📂 Looking for data in: `{output_dir}`")
        
        if not os.path.exists(output_dir):
            st.error(f"❌ Directory not found: {output_dir}\n\nPlease run the aggregation script first:\n```bash\ncd {os.path.dirname(parent_dir)}\npython aggregate_research_metrics.py\n```")
            return
        
        st.success(f"✓ Found aggregation directory")
        
        # Load national-level metrics
        national_file = os.path.join(output_dir, "national_level_aggregation.csv")
        if not os.path.exists(national_file):
            st.error(f"❌ File not found: national_level_aggregation.csv")
            return
        df_national = pd.read_csv(national_file)
        st.success(f"✓ Loaded national metrics ({len(df_national)} rows)")
        
        # Load publication table
        pub_file = os.path.join(output_dir, "publication_national_table.csv")
        df_pub = pd.read_csv(pub_file) if os.path.exists(pub_file) else None
        if df_pub is not None:
            st.success(f"✓ Loaded publication table ({len(df_pub)} models)")
        
        # Load confusion matrix
        cm_file = os.path.join(output_dir, "xgb_confusion_matrix_summary.csv")
        df_cm = pd.read_csv(cm_file) if os.path.exists(cm_file) else None
        if df_cm is not None:
            st.success(f"✓ Loaded confusion matrix ({len(df_cm)} entries)")
        
        # Load feature <-> metric correlations if present
        corr_file = os.path.join(output_dir, "feature_metric_correlations.csv")
        df_corr = pd.read_csv(corr_file) if os.path.exists(corr_file) else None
        if df_corr is not None:
            st.success(f"✓ Loaded feature-metric correlations ({len(df_corr)} rows)")
        
        # Load district rankings
        rankings_file = os.path.join(output_dir, "district_model_rankings.csv")
        df_rankings = pd.read_csv(rankings_file) if os.path.exists(rankings_file) else None
        if df_rankings is not None:
            st.success(f"✓ Loaded district rankings ({len(df_rankings)} districts)")
        
        # Load excluded districts
        excluded_file = os.path.join(output_dir, "excluded_districts.csv")
        df_excluded = pd.read_csv(excluded_file) if os.path.exists(excluded_file) else None
        if df_excluded is not None:
            st.success(f"✓ Loaded excluded districts ({len(df_excluded)} excluded)")

        # Load national SHAP summaries if present
        def _load_shap_file(path):
            if not os.path.exists(path):
                return None
            try:
                df = pd.read_csv(path, index_col=0)
            except Exception:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    return None
            if df.index.name is not None:
                df = df.reset_index().rename(columns={df.columns[0]: "Feature"})
            elif "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Feature"})
            return df

        shap_rf_file = os.path.join(output_dir, "shap_national_Random_Forest.csv")
        shap_xgb_file = os.path.join(output_dir, "shap_national_XGBoost.csv")
        df_shap_rf = _load_shap_file(shap_rf_file)
        df_shap_xgb = _load_shap_file(shap_xgb_file)
        if df_shap_rf is not None:
            st.success(f"✓ Loaded national SHAP importance for Random Forest ({len(df_shap_rf)} features)")
        if df_shap_xgb is not None:
            st.success(f"✓ Loaded national SHAP importance for XGBoost ({len(df_shap_xgb)} features)")
        
        st.divider()
        
        # Display KPI cards
        st.subheader("📊 Key Performance Indicators")
        if df_pub is not None and len(df_pub) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rf_rows = df_pub[df_pub['Model'] == 'Random Forest']
                if len(rf_rows) > 0:
                    # FIXED — safe fallback if column missing
                    rf_nrmse  = rf_rows['NRMSE (mean)'].iloc[0]  if 'NRMSE (mean)' in rf_rows.columns  else np.nan
                    st.metric("RF NRMSE", f"{rf_nrmse:.4f}" if pd.notna(rf_nrmse) else "N/A")
                else:
                    st.metric("RF NRMSE", "N/A")
            
            with col2:
                xgb_rows = df_pub[df_pub['Model'] == 'XGBoost']
                if len(xgb_rows) > 0:
                    xgb_nrmse = xgb_rows['NRMSE (mean)'].iloc[0] if 'NRMSE (mean)' in xgb_rows.columns else np.nan
                    st.metric("XGBoost NRMSE", f"{xgb_nrmse:.4f}" if pd.notna(xgb_nrmse) else "N/A")
                else:
                    st.metric("XGBoost NRMSE", "N/A")
            
            with col3:
                if len(xgb_rows) > 0 and 'R2 (mean)' in df_pub.columns:
                    xgb_r2 = xgb_rows['R2 (mean)'].iloc[0]
                    st.metric("XGBoost R²", f"{xgb_r2:.4f}" if pd.notna(xgb_r2) else "N/A")
                else:
                    st.metric("XGBoost R²", "N/A")
            
            with col4:
                if 'Valid Districts' in df_pub.columns:
                    valid_districts = df_pub['Valid Districts'].iloc[0]
                    st.metric("Valid Districts", int(valid_districts) if pd.notna(valid_districts) else "N/A")
                else:
                    st.metric("Valid Districts", "N/A")
        else:
            st.warning("No publication metrics available")
        
        st.divider()
        
        # Model comparison metrics
        st.subheader("📈 Model Performance Comparison")
        if df_pub is not None and len(df_pub) > 0:
            metrics_to_plot = ["NRMSE (mean)", "R2 (mean)", "NSE (mean)", "KGE (mean)"]
            available_metrics = [m for m in metrics_to_plot if m in df_pub.columns]
            
            if available_metrics:
                fig = make_subplots(
                    rows=1, cols=len(available_metrics),
                    subplot_titles=available_metrics,
                    specs=[[{"type": "bar"} for _ in available_metrics]]
                )
                
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                for idx, metric in enumerate(available_metrics, 1):
                    for model_idx, (_, row) in enumerate(df_pub.iterrows()):
                        model = row.get('Model', f'Model {model_idx}')
                        value = row.get(metric, np.nan)
                        fig.add_trace(
                            go.Bar(x=[model], y=[value], name=model, marker_color=colors[model_idx % len(colors)], showlegend=(idx==1)),
                            row=1, col=idx
                        )
                
                fig.update_layout(height=400, showlegend=True)
                _plotly_chart(fig, key="model_comparison")
        
        # NRMSE comparison
        st.subheader("📉 Normalized RMSE (Lower is Better)")
        if df_pub is not None and len(df_pub) > 0 and "NRMSE (mean)" in df_pub.columns:
            fig_nrmse = go.Figure()
            for _, row in df_pub.iterrows():
                model = row.get('Model', 'Unknown')
                nrmse_val = row.get("NRMSE (mean)", np.nan)
                nrmse_std = row.get("NRMSE (std)", 0) if "NRMSE (std)" in df_pub.columns else 0
                fig_nrmse.add_trace(go.Bar(
                    x=[model],
                    y=[nrmse_val],
                    error_y=dict(type='data', array=[nrmse_std if pd.notna(nrmse_std) else 0]),
                    name=model
                ))
            fig_nrmse.update_layout(title="NRMSE by Model (with std)", height=400, showlegend=False)
            _plotly_chart(fig_nrmse, key="nrmse_comparison")
        
        # R² comparison
        st.subheader("📊 R² Score (Higher is Better)")
        if df_pub is not None and len(df_pub) > 0 and "R2 (mean)" in df_pub.columns:
            fig_r2 = go.Figure()
            for _, row in df_pub.iterrows():
                model = row.get('Model', 'Unknown')
                r2_val = row.get("R2 (mean)", np.nan)
                r2_std = row.get("R2 (std)", 0) if "R2 (std)" in df_pub.columns else 0
                fig_r2.add_trace(go.Bar(
                    x=[model],
                    y=[r2_val],
                    error_y=dict(type='data', array=[r2_std if pd.notna(r2_std) else 0]),
                    name=model
                ))
            fig_r2.update_layout(title="R² by Model (with std)", height=400, showlegend=False)
            _plotly_chart(fig_r2, key="r2_comparison")
        
        # Regime Confusion Matrices (per-model when available)
        st.subheader("🎯 Regime Classification Matrices and Accuracy")

        # Try to use a precomputed confusion matrix CSV (generic)
        if df_cm is not None and len(df_cm) > 0:
            try:
                pivot_cm = df_cm.pivot_table(index="Actual", columns="Pred", values="Count")
                # ensure numeric and fill missing with zeros safely
                pivot_cm = pivot_cm.fillna(0)
                try:
                    pivot_cm = pivot_cm.astype(int)
                except Exception:
                    pivot_cm = pivot_cm.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=pivot_cm.values,
                    x=pivot_cm.columns,
                    y=pivot_cm.index,
                    text=pivot_cm.values,
                    texttemplate="%{text}",
                    colorscale="Blues"
                ))
                fig_cm.update_layout(title="Confusion Matrix (National Totals)", height=400)
                _plotly_chart(fig_cm, key="confusion_matrix")
                st.dataframe(pivot_cm, use_container_width=True)
            except Exception as cm_err:
                st.warning(f"Could not render confusion matrix heatmap: {cm_err}")
                st.dataframe(df_cm, use_container_width=True)

        # If a cleaned dataset exists, attempt to compute per-model confusion + accuracy
        clean_file = os.path.join(output_dir, "cleaned_dataset_used_for_aggregation.csv")
        df_clean = pd.read_csv(clean_file) if os.path.exists(clean_file) else None
        df_regime_accuracy = None
        df_regime_cm_by_model = None
        if df_clean is not None:
            cm_cols = [c for c in df_clean.columns if 'cm__' in c]
            if cm_cols:
                rows = []
                for c in cm_cols:
                    try:
                        left, right = c.split('cm__', 1)
                        model = left.rstrip('_').strip()
                        # right looks like 'Actual Low__Pred Low' -> extract labels
                        if '__Pred ' in right:
                            actual_part, pred_part = right.split('__Pred ', 1)
                        else:
                            parts = right.split('__Pred')
                            actual_part = parts[0]
                            pred_part = parts[-1]
                        actual = actual_part.replace('Actual', '').replace('__', '').strip().lstrip(':').strip()
                        pred = pred_part.replace('__', '').strip()
                        vals = pd.to_numeric(df_clean[c], errors='coerce').fillna(0)
                        count = vals.sum()
                        rows.append({'Model': model or 'model', 'Actual': actual or 'Unknown', 'Pred': pred or 'Unknown', 'Count': int(count)})
                    except Exception:
                        continue
                if rows:
                    df_regime_cm_by_model = pd.DataFrame(rows)
                    try:
                        df_regime_cm_by_model.to_csv(os.path.join(output_dir, 'regime_confusion_by_model.csv'), index=False)
                    except Exception:
                        pass

                    # compute per-model accuracy/precision/recall/f1 (macro)
                    metrics_rows = []
                    for model, g in df_regime_cm_by_model.groupby('Model'):
                        pivot = g.pivot_table(index='Actual', columns='Pred', values='Count', fill_value=0)
                        total = pivot.values.sum()
                        correct = 0
                        for lab in pivot.index:
                            if lab in pivot.columns:
                                correct += pivot.at[lab, lab]
                        accuracy = float(correct) / float(total) if total > 0 else np.nan
                        precisions = []
                        recalls = []
                        for lab in pivot.index:
                            tp = pivot.at[lab, lab] if lab in pivot.columns else 0
                            fp = pivot[lab].sum() - tp if lab in pivot.columns else 0
                            fn = pivot.loc[lab].sum() - tp
                            prec = float(tp) / (tp + fp) if (tp + fp) > 0 else np.nan
                            rec = float(tp) / (tp + fn) if (tp + fn) > 0 else np.nan
                            precisions.append(prec)
                            recalls.append(rec)
                        precision_macro = float(np.nanmean([p for p in precisions if pd.notna(p)])) if any(pd.notna(precisions)) else np.nan
                        recall_macro = float(np.nanmean([r for r in recalls if pd.notna(r)])) if any(pd.notna(recalls)) else np.nan
                        f1s = [2 * p * r / (p + r) if (pd.notna(p) and pd.notna(r) and (p + r) > 0) else np.nan for p, r in zip(precisions, recalls)]
                        f1_macro = float(np.nanmean([x for x in f1s if pd.notna(x)])) if any(pd.notna(f1s)) else np.nan
                        metrics_rows.append({'Model': model, 'Accuracy': accuracy, 'Precision (macro)': precision_macro, 'Recall (macro)': recall_macro, 'F1 (macro)': f1_macro, 'Total': int(total)})
                    df_regime_accuracy = pd.DataFrame(metrics_rows)
                    try:
                        df_regime_accuracy.to_csv(os.path.join(output_dir, 'regime_accuracy_table.csv'), index=False)
                    except Exception:
                        pass

        # Render per-model confusion heatmaps and accuracy table if computed
        if df_regime_cm_by_model is not None and len(df_regime_cm_by_model) > 0:
            st.write("**Per-model confusion aggregates (computed from cleaned dataset)**")
            for model, g in df_regime_cm_by_model.groupby('Model'):
                st.markdown(f"**Model: {model}**")
                pivot = g.pivot_table(index='Actual', columns='Pred', values='Count')
                pivot = pivot.fillna(0)
                try:
                    pivot = pivot.astype(int)
                except Exception:
                    pivot = pivot.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                fig_m = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Blues', text=pivot.values, texttemplate="%{text}"))
                fig_m.update_layout(height=350, title=f"Confusion: {model}")
                _plotly_chart(fig_m, key=f"cm_{model}")
                st.dataframe(pivot, use_container_width=True)

        if df_regime_accuracy is not None and len(df_regime_accuracy) > 0:
            st.write("**Regime classification accuracy summary**")
            # Sanitize numeric columns to avoid Arrow overflow when Streamlit converts to Arrow
            df_ra = df_regime_accuracy.sort_values('Accuracy', ascending=False).reset_index(drop=True).copy()
            for col in df_ra.columns:
                # attempt to coerce any column to numeric; if successful, keep as float and clean extremes
                temp = pd.to_numeric(df_ra[col], errors='coerce')
                if temp.notna().sum() == 0:
                    # nothing numeric in this column
                    continue
                # convert to float for safe Arrow conversion
                try:
                    temp = temp.astype(float)
                except Exception:
                    temp = pd.to_numeric(temp, errors='coerce')

                # remove infinities and non-finite
                temp[~np.isfinite(temp)] = np.nan

                # cap/clean very large magnitudes that may overflow Arrow (int64 limit ~9e18)
                # use a conservative threshold to detect erroneous huge counts
                huge_thresh = 1e12
                try:
                    temp[temp.abs() > huge_thresh] = np.nan
                except Exception:
                    pass

                df_ra[col] = temp

            st.dataframe(df_ra, use_container_width=True)

        # National SHAP feature importance
        st.subheader("🧠 National SHAP feature importance")
        if df_shap_rf is None and df_shap_xgb is None:
            st.write("No national SHAP files were found. Run the aggregation script to generate shap_national_Random_Forest.csv and shap_national_XGBoost.csv.")
        else:
            if df_shap_rf is not None:
                st.markdown("#### Random Forest")
                st.dataframe(df_shap_rf.sort_values('mean_abs_shap', ascending=False).head(20), use_container_width=True)
            if df_shap_xgb is not None:
                st.markdown("#### XGBoost")
                st.dataframe(df_shap_xgb.sort_values('mean_abs_shap', ascending=False).head(20), use_container_width=True)

        # Feature ↔ Metric correlations
        st.subheader("🔗 Feature vs. Metric Correlations")
        if df_corr is None:
            st.write("No precomputed feature-metric correlations found. Run the aggregation script to generate `feature_metric_correlations.csv`.")
        else:
            # show top absolute correlations
            df_corr['absR'] = df_corr['WeightedPearsonR'].abs()
            topk = st.slider('Top features to show (by |r|)', min_value=5, max_value=100, value=20)
            df_top = df_corr.sort_values('absR', ascending=False).head(topk)
            st.dataframe(df_top[['Feature','Model','Metric','WeightedPearsonR','N_valid']].reset_index(drop=True), use_container_width=True)
            # allow user to filter by model and metric
            st.markdown('**Filter by model / metric**')
            models_available = sorted(df_corr['Model'].unique())
            metrics_available = sorted(df_corr['Metric'].unique())
            sel_model = st.selectbox('Model', options=['ALL'] + models_available)
            sel_metric = st.selectbox('Metric', options=['ALL'] + metrics_available)
            mask = pd.Series(True, index=df_corr.index)
            if sel_model != 'ALL':
                mask &= df_corr['Model'] == sel_model
            if sel_metric != 'ALL':
                mask &= df_corr['Metric'] == sel_metric
            df_filt = df_corr[mask].copy()
            if df_filt.empty:
                st.write('No feature correlations for this selection.')
            else:
                df_filt['absR'] = df_filt['WeightedPearsonR'].abs()
                st.dataframe(df_filt.sort_values('absR', ascending=False).reset_index(drop=True), use_container_width=True)

        
        # Model ranking wins
        st.subheader("🏆 Model Win Frequencies (by District)")
        if df_rankings is not None and len(df_rankings) > 0 and "rank_1" in df_rankings.columns:
            rank1_counts = df_rankings["rank_1"].value_counts()
            if len(rank1_counts) > 0:
                fig_wins = go.Figure()
                for model, count in rank1_counts.items():
                    total_valid = len(df_rankings.dropna(subset=["rank_1"]))
                    pct = 100 * count / total_valid if total_valid > 0 else 0
                    fig_wins.add_trace(go.Bar(x=[model], y=[count], name=model, text=f"{pct:.1f}%", textposition="auto"))
                fig_wins.update_layout(title="Districts Where Each Model Ranks #1", height=400, showlegend=False)
                _plotly_chart(fig_wins, key="model_wins")

        # IsolationForest / Anomaly density summaries
        anom_file = os.path.join(output_dir, 'anomaly_density_by_district.csv')
        df_anom = None
        if os.path.exists(anom_file):
            try:
                df_anom = pd.read_csv(anom_file)
                st.success(f"✓ Loaded anomaly density file ({len(df_anom)} districts)")
            except Exception as e:
                st.warning(f"Could not read anomaly density file: {e}")

        if df_anom is None and df_clean is not None:
            # fallback: derive anomaly density from cleaned dataset if explicit anomaly file is missing
            candidate_cols = [c for c in df_clean.columns if c.lower().startswith('if__anomaly_density') or c.lower().startswith('if__anomaly_count')]
            if 'district' in df_clean.columns and candidate_cols:
                df_anom = df_clean[['state', 'district'] + [c for c in candidate_cols if c in df_clean.columns]].copy()
                st.info("Fallback: using cleaned aggregation dataset to derive anomaly density summaries.")

        if df_anom is not None:
            try:
                st.subheader("🧭 Anomaly Density (district-level)")
                possible_cols = [c for c in df_anom.columns if 'anomaly' in c.lower() or 'isolation' in c.lower() or 'score' in c.lower()]
                if len(possible_cols) == 0:
                    st.write("No anomaly density column detected in anomaly data.")
                else:
                    col = possible_cols[0]
                    fig_ad = go.Figure()
                    fig_ad.add_trace(go.Histogram(x=df_anom[col].dropna(), nbinsx=40))
                    fig_ad.update_layout(title=f"Distribution of {col}", height=350)
                    _plotly_chart(fig_ad, key="anom_hist")
                    st.write("Top anomalous districts")
                    if 'district' in df_anom.columns:
                        top = df_anom.sort_values(by=col, ascending=False).head(20)
                        st.dataframe(top[['district', 'state', col]] if 'state' in df_anom.columns else top[['district', col]], use_container_width=True)
                    state_col = None
                    for cand in ['state', 'State', 'STATE']:
                        if cand in df_anom.columns:
                            state_col = cand
                            break
                    if state_col is not None:
                        df_state_anom = df_anom.groupby(state_col)[col].mean().reset_index().rename(columns={col: 'anomaly_mean'})
                        try:
                            df_state_anom.to_csv(os.path.join(output_dir, 'anomaly_density_by_state.csv'), index=False)
                        except Exception:
                            pass
                        st.subheader("Anomaly density by state (mean)")
                        fig_state = go.Figure(go.Bar(x=df_state_anom[state_col], y=df_state_anom['anomaly_mean']))
                        fig_state.update_layout(height=350)
                        _plotly_chart(fig_state, key="anom_state")
            except Exception as e:
                st.warning(f"Could not load anomaly data: {e}")
        else:
            st.warning("No anomaly density output was found. Run the aggregation script or ensure anomaly density columns exist in the cleaned dataset.")

        st.divider()
        
        # Summary statistics
        st.subheader("📋 Aggregation Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            if df_excluded is not None:
                st.metric("Districts Excluded", len(df_excluded))
            if df_national is not None and len(df_national) > 0 and "n_districts" in df_national.columns:
                st.metric("Valid Districts", int(df_national["n_districts"].iloc[0]))
        with col2:
            if df_excluded is not None and len(df_excluded) > 0:
                st.write("**Top Exclusion Reasons:**")
                if "exclude_reasons" in df_excluded.columns:
                    reasons = df_excluded["exclude_reasons"].value_counts()
                    for reason, count in reasons.head(3).items():
                        st.write(f"  • {reason}: {count}")
        with col3:
            st.write("**Publication Metrics:**")
            if df_pub is not None:
                st.dataframe(df_pub[["Model", "NRMSE (mean)", "R2 (mean)", "Valid Districts"]], use_container_width=True)
        
        st.divider()
        
        # Detailed metrics table
        st.subheader("📊 Detailed National Metrics")
        st.dataframe(df_national, use_container_width=True)
        
        st.divider()
        
        # Download buttons
        st.subheader("💾 Download Outputs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if df_national is not None:
                st.download_button(
                    "📊 National Metrics",
                    data=df_national.to_csv(index=False).encode('utf-8'),
                    file_name="national_level_aggregation.csv",
                    mime="text/csv"
                )
        with col2:
            if df_pub is not None:
                st.download_button(
                    "📈 Publication Table",
                    data=df_pub.to_csv(index=False).encode('utf-8'),
                    file_name="publication_national_table.csv",
                    mime="text/csv"
                )
        with col3:
            if df_cm is not None:
                st.download_button(
                    "🎯 Confusion Matrix",
                    data=df_cm.to_csv(index=False).encode('utf-8'),
                    file_name="xgb_confusion_matrix_summary.csv",
                    mime="text/csv"
                )
        with col4:
            if df_rankings is not None:
                st.download_button(
                    "🏆 Model Rankings",
                    data=df_rankings.to_csv(index=False).encode('utf-8'),
                    file_name="district_model_rankings.csv",
                    mime="text/csv"
                )
        # extra downloads (regime accuracy, anomaly state)
        cols_extra = st.columns(3)
        with cols_extra[0]:
            ra_file = os.path.join(output_dir, 'regime_accuracy_table.csv')
            if os.path.exists(ra_file):
                st.download_button('📈 Regime Accuracy', data=open(ra_file,'rb').read(), file_name='regime_accuracy_table.csv', mime='text/csv')
        with cols_extra[1]:
            rcm_file = os.path.join(output_dir, 'regime_confusion_by_model.csv')
            if os.path.exists(rcm_file):
                st.download_button('🎯 Regime Confusions (by model)', data=open(rcm_file,'rb').read(), file_name='regime_confusion_by_model.csv', mime='text/csv')
        with cols_extra[2]:
            an_state_file = os.path.join(output_dir, 'anomaly_density_by_state.csv')
            if os.path.exists(an_state_file):
                st.download_button('🧭 Anomaly by State', data=open(an_state_file,'rb').read(), file_name='anomaly_density_by_state.csv', mime='text/csv')
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        st.error(f"❌ Error: {error_type}: {error_msg}")
        st.error(f"Traceback:\n```\n{traceback.format_exc()}\n```")


def main():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #ffffff;
            color: #000000;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        }
        [data-testid="stDataFrame"] {
            background: #ffffff;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            color: #000000;
        }
        .hero {
            padding: 1.1rem 1.3rem;
            border-radius: 18px;
            background: #ffffff;
            color: #000000;
            margin-bottom: 1rem;
            border: 1px solid #e5e5e5;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.05);
        }
        .subtle { opacity: 0.78; }
        .section-card {
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.04);
        }
        .stButton button,
        .stSelectbox [data-baseweb="select"],
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSlider {
            background: #ffffff !important;
            color: #000000 !important;
        }
        h1, h2, h3, h4, p, label, .stMarkdown, .stCaption {
            color: #000000;
        }
        .css-1kyxreq { background: #ffffff !important; }
        .css-1v0mbdj { color: #000000 !important; }
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

        state_choices = fetch_state_choices(backend_url)
        if state_choices:
            default_state = "Odisha" if "Odisha" in state_choices else state_choices[0]
            state = st.selectbox("State", state_choices, index=state_choices.index(default_state))
        else:
            st.warning("State list could not be loaded from the backend. Using manual entry.")
            state = st.text_input("State", "Odisha")

        district_choices = fetch_district_choices(backend_url, state)
        if district_choices:
            default_district = "Baleshwar" if "Baleshwar" in district_choices else district_choices[0]
            district = st.selectbox("District", district_choices, index=district_choices.index(default_district))
        else:
            st.warning("District list could not be loaded from the backend for the selected state. Using manual entry.")
            district = st.text_input("District", "Baleshwar")

        agency_choices = fetch_agency_choices(backend_url)
        if agency_choices:
            default_agency = "CGWB" if "CGWB" in agency_choices else agency_choices[0]
            agency = st.selectbox("Agency", agency_choices, index=agency_choices.index(default_agency))
        else:
            st.warning("Agency list could not be loaded from the backend. Using manual entry.")
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

        home_tab, research_tab, national_tab = st.tabs(["Home", "Research", "National Aggregation"])

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
            render_dataset_analysis_panel(state, district, trained["featured"], trained, forecast_df, build_benchmark_aggregates(load_benchmark_runs()))

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
            render_alert_tables(load_benchmark_runs())

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

        with national_tab:
            st.subheader("National-Level Aggregation Metrics")
            st.caption("All India aggregated metrics across models with comparative visualizations.")
            render_national_aggregation_dashboard()

        st.success("Dashboard ready: map, heatmaps, explainability, and research validation are available in a simplified layout.")

    else:
        st.info("Choose your parameters in the sidebar and run the analysis.")


if __name__ == "__main__":
    main()