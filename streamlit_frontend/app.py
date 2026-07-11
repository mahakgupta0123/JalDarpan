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
RESEARCH_COLORS = ["#0f4c81", "#2b8cbe", "#f28e2b", "#2ca02c", "#d95f02", "#6a3d9a"]
NATIONAL_FIGURE_DIR = os.path.join(parent_dir, "aggregation_outputs", "research_figures")

INDIA_STATE_CENTROIDS = {
    "andhra pradesh": (16.50, 80.64),
    "arunachal pradesh": (27.10, 93.62),
    "assam": (26.20, 92.93),
    "bihar": (25.59, 85.13),
    "chhattisgarh": (21.25, 81.63),
    "goa": (15.49, 73.83),
    "gujarat": (23.02, 72.57),
    "haryana": (29.06, 76.09),
    "himachal pradesh": (31.10, 77.17),
    "jharkhand": (23.34, 85.31),
    "karnataka": (12.97, 77.59),
    "kerala": (8.52, 76.94),
    "madhya pradesh": (23.26, 77.41),
    "maharashtra": (19.08, 72.88),
    "manipur": (24.82, 93.94),
    "meghalaya": (25.57, 91.88),
    "mizoram": (23.73, 92.72),
    "nagaland": (25.67, 94.11),
    "odisha": (20.27, 85.84),
    "punjab": (31.63, 74.87),
    "rajasthan": (26.91, 75.79),
    "sikkim": (27.33, 88.61),
    "tamil nadu": (13.08, 80.27),
    "telangana": (17.38, 78.49),
    "tripura": (23.83, 91.29),
    "uttar pradesh": (26.85, 80.95),
    "uttarakhand": (30.32, 78.03),
    "west bengal": (22.57, 88.36),
    "andaman and nicobar islands": (11.67, 92.73),
    "chandigarh": (30.74, 76.79),
    "dadra and nagar haveli and daman and diu": (20.40, 72.95),
    "delhi": (28.61, 77.21),
    "jammu and kashmir": (34.08, 74.80),
    "ladakh": (34.15, 77.58),
    "lakshadweep": (10.57, 72.64),
    "puducherry": (11.94, 79.81),
}


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
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )


def _save_research_figure(fig, filename_base):
    try:
        os.makedirs(NATIONAL_FIGURE_DIR, exist_ok=True)
        png_path = os.path.join(NATIONAL_FIGURE_DIR, f"{filename_base}.png")
        html_path = os.path.join(NATIONAL_FIGURE_DIR, f"{filename_base}.html")
        try:
            fig.write_image(png_path, scale=2)
            return png_path
        except Exception:
            fig.write_html(html_path, include_plotlyjs="cdn")
            return html_path
    except Exception:
        return None


def _state_center(state_name):
    key = _safe_text(state_name, "").strip().lower()
    if not key:
        return None
    if key in INDIA_STATE_CENTROIDS:
        return INDIA_STATE_CENTROIDS[key]
    for known_state, coords in INDIA_STATE_CENTROIDS.items():
        if known_state in key or key in known_state:
            return coords
    return None


def apply_research_style(fig, *, title=None, height=420, showlegend=True, x_title=None, y_title=None, tickangle=0, legend_title=None, colorway=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=48, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#10313e"),
        title=dict(text=title or "", x=0.05, xanchor="left", font=dict(size=18, color="#10313e", family="Arial")),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#000000"),
            title_text=legend_title or "",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        colorway=colorway or RESEARCH_COLORS,
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title, title_font=dict(color="#000000"), showgrid=False, zeroline=False, showline=True, linecolor="#d0d7de", ticks="outside", tickfont=dict(size=11, color="#4b5c68"))
    else:
        fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor="#d0d7de", ticks="outside", tickfont=dict(size=11, color="#4b5c68"))
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, title_font=dict(color="#000000"), showgrid=True, gridcolor="#e9eff3", zeroline=False, showline=True, linecolor="#d0d7de", ticks="outside", tickfont=dict(size=11, color="#4b5c68"))
    else:
        fig.update_yaxes(showgrid=True, gridcolor="#e9eff3", zeroline=False, showline=True, linecolor="#d0d7de", ticks="outside", tickfont=dict(size=11, color="#4b5c68"))
    if tickangle:
        fig.update_xaxes(tickangle=tickangle)
    if not showlegend:
        fig.update_layout(showlegend=False)

    for trace in fig.data:
        trace_type = getattr(trace, "type", None)
        if trace_type == "scattermapbox":
            continue
        try:
            trace.update(marker=dict(line=dict(color="#0f4c81", width=0.7)))
        except Exception:
            continue
    return fig


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

    st.markdown("### Feature engineering overview")
    st.write(
        f"- Generated {len(trained['feature_cols'])} engineered features from the cleaned time series."
    )
    st.write("- Normalized the most recent groundwater value relative to the selected district range.")
    st.write("- Used rolling means, rolling standard deviations, lag values, differences, and seasonal encodings.")


def render_shap_explanation_panel(feature_cols, shap_values):
    shap_info = describe_xai_meaning(feature_cols, shap_values)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("SHAP driver plot")
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False).head(10)
        shap_fig = go.Figure(go.Bar(x=shap_importance.values, y=shap_importance.index, orientation="h"))
        shap_fig = apply_research_style(
            shap_fig,
            title="Top SHAP drivers",
            height=420,
            x_title="Mean absolute SHAP value",
            y_title="Feature",
            showlegend=False,
        )
        _plotly_chart(shap_fig, key="district_shap_chart")
    with right:
        st.subheader("Plain-language explanation")
        st.write(shap_info["summary_text"])
        st.write(shap_info["plain_language"])
        st.caption("This explanation sits next to the SHAP chart so users can see both the driver ranking and a simple interpretation together.")


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
    )
    return apply_research_style(fig, title="Geographical view", height=500, showlegend=False)


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
    return apply_research_style(fig, title="Anomaly score over time", height=380, x_title="Time", y_title="Anomaly score")


def build_forecast_figure(featured, forecast_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=featured.index,
            y=featured["groundwater_level"],
            name="Actual",
            line=dict(color="#2166ac", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast"],
            name="Predicted",
            line=dict(color="#b2182b", width=3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
            y=forecast_df["upper"].tolist() + forecast_df["lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(178, 24, 43, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Uncertainty band",
            hoverinfo="skip",
        )
    )
    return apply_research_style(
        fig,
        title="Actual vs predicted groundwater level",
        height=500,
        x_title="Time",
        y_title="Groundwater level",
    )


def build_national_metric_figure(df_pub):
    preferred_metric_groups = [
        ["RMSE_mean", "RMSE (mean)", "RMSE mean"],
        ["MAE_mean", "MAE (mean)", "MAE mean"],
        ["NRMSE_mean", "NRMSE (mean)", "NRMSE mean"],
        ["MAPE_mean", "MAPE (mean)", "MAPE mean"],
    ]
    preferred_skill_groups = [
        ["R2_mean", "R2 (mean)", "R2 mean"],
        ["mNSE_mean", "mNSE (mean)", "mNSE mean"],
        ["KGE_mean", "KGE (mean)", "KGE mean"],
    ]
    if df_pub is None or df_pub.empty:
        return _build_placeholder_heatmap(
            "Model performance figures unavailable",
            "No publication table was found in the aggregation outputs.",
        )

    def select_column(candidates):
        for candidate in candidates:
            if candidate in df_pub.columns:
                return candidate
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Error metrics", "Skill metrics"),
        horizontal_spacing=0.12,
    )

    error_metrics = [select_column(candidates) for candidates in preferred_metric_groups]
    skill_metrics = [select_column(candidates) for candidates in preferred_skill_groups]

    if not any(error_metrics + skill_metrics):
        return _build_placeholder_heatmap(
            "National performance chart unavailable",
            "The aggregation file does not contain the expected national metric columns.",
        )

    model_labels = df_pub["Model"] if "Model" in df_pub.columns else [f"Model {idx + 1}" for idx in range(len(df_pub))]

    for idx, metric in enumerate(error_metrics):
        if metric is None:
            continue
        fig.add_trace(
            go.Bar(
                x=model_labels,
                y=df_pub[metric],
                name=metric.replace("_", " "),
                marker_color=RESEARCH_COLORS[idx % len(RESEARCH_COLORS)],
                opacity=0.92,
            ),
            row=1,
            col=1,
        )

    for idx, metric in enumerate(skill_metrics):
        if metric is None:
            continue
        fig.add_trace(
            go.Bar(
                x=model_labels,
                y=df_pub[metric],
                name=metric.replace("_", " "),
                marker_color=RESEARCH_COLORS[(idx + len(error_metrics)) % len(RESEARCH_COLORS)],
                opacity=0.92,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(barmode="group")

    fig = apply_research_style(
        fig,
        title="National model performance comparison",
        height=520,
        x_title="Model",
        y_title="Score",
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.18,
            x=1.0,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#000000"),
        )
    )
    return fig


def build_national_anomaly_map(df_anom_state):
    if df_anom_state is None or df_anom_state.empty:
        return _build_placeholder_heatmap(
            "India anomaly map unavailable",
            "No state-level anomaly density file was found.",
        )

    df_map = df_anom_state.copy()
    state_col = _safe_column(df_map, ["state", "State", "STATE"])
    density_col = _safe_column(df_map, ["IF__anomaly_density", "anomaly_mean", "anomaly_density", "anomaly_density_mean"])
    if state_col is None or density_col is None:
        return _build_placeholder_heatmap(
            "India anomaly map unavailable",
            "State or anomaly density columns were missing.",
        )

    points = []
    for _, row in df_map.iterrows():
        center = _state_center(row[state_col])
        if center is None:
            continue
        density = pd.to_numeric(pd.Series([row[density_col]]), errors="coerce").iloc[0]
        if pd.isna(density):
            continue
        points.append(
            {
                "state": row[state_col],
                "lat": center[0],
                "lon": center[1],
                "density": float(density),
            }
        )

    if not points:
        return _build_placeholder_heatmap(
            "India anomaly map unavailable",
            "No states matched the built-in India centroid lookup.",
        )

    df_points = pd.DataFrame(points).sort_values("density", ascending=False)
    max_density = float(df_points["density"].max()) if len(df_points) else 1.0
    size_scale = 10 if max_density <= 0 else 12 + 18 * (df_points["density"] / max_density)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=df_points["lon"],
            lat=df_points["lat"],
            text=df_points["state"],
            customdata=np.array(df_points[["density"]]),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=size_scale,
                color=df_points["density"],
                colorscale="YlOrRd",
                cmin=float(df_points["density"].min()),
                cmax=float(df_points["density"].max()),
                line=dict(color="#1f2933", width=0.7),
                opacity=0.92,
                colorbar=dict(title="Anomaly density"),
            ),
            hovertemplate="%{text}<br>Anomaly density: %{customdata[0]:.3f}<extra></extra>",
            name="State anomaly density",
        )
    )
    fig.update_geos(
        scope="asia",
        showcountries=True,
        countrycolor="#c8d2da",
        showland=True,
        landcolor="#f5f8fb",
        showocean=True,
        oceancolor="#eaf3f9",
        showlakes=True,
        lakecolor="#eaf3f9",
        center=dict(lat=22.5, lon=80.5),
        fitbounds="locations",
        projection_type="equirectangular",
        lataxis=dict(range=[6, 38]),
        lonaxis=dict(range=[67, 98]),
    )
    return apply_research_style(
        fig,
        title="India anomaly density map",
        height=680,
        showlegend=False,
    )


def build_national_anomaly_bar(df_anom_state):
    if df_anom_state is None or df_anom_state.empty:
        return _build_placeholder_heatmap(
            "Anomalous states figure unavailable",
            "No state-level anomaly density file was found.",
        )
    state_col = _safe_column(df_anom_state, ["state", "State", "STATE"])
    density_col = _safe_column(df_anom_state, ["IF__anomaly_density", "anomaly_mean", "anomaly_density", "anomaly_density_mean"])
    if state_col is None or density_col is None:
        return _build_placeholder_heatmap(
            "Anomalous states figure unavailable",
            "State or anomaly density columns were missing.",
        )
    summary = (
        df_anom_state[[state_col, density_col]]
        .assign(**{density_col: pd.to_numeric(df_anom_state[density_col], errors="coerce")})
        .dropna()
        .sort_values(density_col, ascending=False)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary[state_col],
            y=summary[density_col],
            marker=dict(
                color=summary[density_col],
                colorscale="YlOrRd",
                line=dict(color="#1f2933", width=0.4),
            ),
            text=summary[density_col].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            name="Anomaly density",
        )
    )
    fig.update_xaxes(tickangle=-35)
    return apply_research_style(
        fig,
        title="Most anomalous states",
        height=420,
        x_title="State",
        y_title="Anomaly density",
        showlegend=False,
    )


def build_national_shap_bar(df_clean, model="XGBoost", top_n=15):
    shap_prefix = f"SHAP__{model}__"
    shap_cols = [col for col in df_clean.columns if isinstance(col, str) and col.startswith(shap_prefix) and "top" not in col]
    if not shap_cols:
        return _build_placeholder_heatmap(
            f"{model} SHAP barplot unavailable",
            "No SHAP contribution columns were found in the national aggregation dataset.",
        )

    rows = pd.to_numeric(df_clean.get("rows", pd.Series(1, index=df_clean.index)), errors="coerce").fillna(1.0)
    feature_scores = []
    for col in shap_cols:
        feature_name = col.replace(shap_prefix, "")
        values = pd.to_numeric(df_clean[col], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        mean_abs = np.average(np.abs(values[valid]), weights=rows[valid])
        feature_scores.append((feature_name, float(mean_abs)))

    if not feature_scores:
        return _build_placeholder_heatmap(
            f"{model} SHAP barplot unavailable",
            "No valid SHAP values were available for aggregation.",
        )

    summary = pd.DataFrame(feature_scores, columns=["Feature", "MeanAbsSHAP"])
    summary = summary.sort_values("MeanAbsSHAP", ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            x=summary["MeanAbsSHAP"],
            y=summary["Feature"],
            orientation="h",
            marker=dict(
                color=summary["MeanAbsSHAP"],
                colorscale="Blues",
                line=dict(color="#1f2933", width=0.5),
            ),
            text=summary["MeanAbsSHAP"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            name="Mean |SHAP|",
        )
    )
    return apply_research_style(
        fig,
        title=f"National SHAP barplot - {model}",
        height=500,
        x_title="Mean absolute SHAP",
        y_title="Feature",
        showlegend=False,
    )


def build_national_shap_beeswarm(df_clean, model="XGBoost", top_n=12):
    shap_prefix = f"SHAP__{model}__"
    shap_cols = [col for col in df_clean.columns if isinstance(col, str) and col.startswith(shap_prefix) and "top" not in col]
    if not shap_cols:
        return _build_placeholder_heatmap(
            f"{model} SHAP beeswarm unavailable",
            "No SHAP contribution columns were found in the national aggregation dataset.",
        )

    rows = pd.to_numeric(df_clean.get("rows", pd.Series(1, index=df_clean.index)), errors="coerce").fillna(1.0)
    feature_scores = []
    for col in shap_cols:
        feature_name = col.replace(shap_prefix, "")
        values = pd.to_numeric(df_clean[col], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        mean_abs = np.average(np.abs(values[valid]), weights=rows[valid])
        feature_scores.append((feature_name, float(mean_abs), values))

    if not feature_scores:
        return _build_placeholder_heatmap(
            f"{model} SHAP beeswarm unavailable",
            "No valid SHAP values were available for aggregation.",
        )

    feature_scores = sorted(feature_scores, key=lambda item: item[1], reverse=True)[:top_n]
    fig = go.Figure()
    color_range = []
    for rank, (feature_name, _, values) in enumerate(feature_scores):
        valid_values = pd.to_numeric(values, errors="coerce").dropna()
        if valid_values.empty:
            continue
        jitter = (np.random.RandomState(42 + rank).rand(len(valid_values)) - 0.5) * 0.42
        y_positions = np.full(len(valid_values), rank) + jitter
        color_range.extend(valid_values.tolist())
        marker_kwargs = dict(
            size=6,
            color=valid_values,
            colorscale="RdBu",
            cmin=-float(np.nanmax(np.abs(valid_values))) if np.isfinite(np.nanmax(np.abs(valid_values))) else None,
            cmax=float(np.nanmax(np.abs(valid_values))) if np.isfinite(np.nanmax(np.abs(valid_values))) else None,
            cmid=0,
            opacity=0.8,
            showscale=(rank == 0),
            line=dict(color="rgba(255,255,255,0.25)", width=0.4),
        )
        if rank == 0:
            marker_kwargs["colorbar"] = dict(title="SHAP value")
        fig.add_trace(
            go.Scatter(
                x=valid_values,
                y=y_positions,
                mode="markers",
                name=feature_name,
                marker=marker_kwargs,
                hovertemplate=f"{feature_name}<br>SHAP: %{{x:.4f}}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(feature_scores))),
        ticktext=[item[0] for item in feature_scores],
        autorange="reversed",
    )
    if color_range:
        limit = float(np.nanmax(np.abs(color_range))) if np.isfinite(np.nanmax(np.abs(color_range))) else None
        if limit is not None:
            fig.update_xaxes(range=[-limit, limit])
    return apply_research_style(
        fig,
        title=f"National SHAP beeswarm - {model}",
        height=540,
        x_title="SHAP value",
        y_title="Feature",
        showlegend=False,
    )


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
    fig.update_xaxes(tickangle=-45)
    return apply_research_style(fig, title="Daily groundwater level", height=260, x_title="Date", y_title="Groundwater level")


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
    return apply_research_style(fig, title=title, height=520, showlegend=False)


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
    return apply_research_style(fig, title="Engineered feature correlation matrix", height=520, x_title="Feature", y_title="Feature")




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
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
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


def build_district_model_metrics_table(trained):
    metrics_frame = trained.get("metrics") if isinstance(trained, dict) else None
    if metrics_frame is None or getattr(metrics_frame, "empty", True):
        return pd.DataFrame(columns=["Model", "RMSE", "MAE", "MAPE", "R2", "NSE", "NRMSE", "Regime Acc", "Regime Coverage", "Regime Majority Share", "Regime Valid"])

    table = metrics_frame.reset_index()
    if "index" in table.columns:
        table = table.rename(columns={"index": "Model"})
    elif "Model" not in table.columns:
        table.insert(0, "Model", [str(idx) for idx in range(len(table))])

    display_cols = ["Model"]
    for column in ["RMSE", "MAE", "MAPE", "R2", "NSE", "NRMSE", "Regime Acc", "Regime Coverage", "Regime Majority Share", "Regime Valid"]:
        if column in table.columns:
            display_cols.append(column)

    return table[display_cols].copy()


def load_local_district_reference():
    ref_path = os.path.join(parent_dir, "aggregation_outputs", "cleaned_dataset_used_for_aggregation.csv")
    if not os.path.exists(ref_path):
        return pd.DataFrame()

    frame = pd.read_csv(ref_path)
    if frame.empty:
        return frame

    for column in ["state", "district"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
            frame[f"{column}_norm"] = frame[column].str.lower()

    # Deduplicate by district so the top-10 tables show unique districts only.
    if "state" in frame.columns and "district" in frame.columns:
        numeric_cols = {
            "IF__anomaly_density": "max",
            "IF__anomaly_count": "max",
            "gwl_min": "min",
            "gwl_max": "max",
        }
        agg_cols = {col: func for col, func in numeric_cols.items() if col in frame.columns}
        if agg_cols:
            frame = frame.groupby(["state", "district"], as_index=False).agg(agg_cols)
    return frame


def render_district_level_home_tab(state, district, featured, forecast_df, trained, location_context, shap_values, feature_cols):
    latest_level = float(featured["groundwater_level"].iloc[-1]) if not featured.empty else np.nan
    recent_window = featured["groundwater_level"].tail(min(14, len(featured))) if not featured.empty else pd.Series(dtype=float)
    recent_change = float(recent_window.iloc[-1] - recent_window.iloc[0]) if len(recent_window) > 1 else 0.0
    forecasted_level = float(forecast_df["forecast"].iloc[0]) if forecast_df is not None and not forecast_df.empty else np.nan
    anomaly_density_pct = float(featured["anomaly_flag"].mean() * 100) if not featured.empty else np.nan
    st.caption("This view emphasizes the current district state, the short-term forecast, and the local district context. The district rankings below are sourced from the local aggregation cache for a broader comparison.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current groundwater level", f"{latest_level:.2f}", delta=f"{recent_change:.2f} over recent window")
    with col2:
        st.metric("Forecasted level", f"{forecasted_level:.2f}", delta=f"{forecasted_level - latest_level:.2f} vs current")
    with col3:
        st.metric("Anomaly density", f"{anomaly_density_pct:.2f}%", delta="flagged readings")

    st.divider()

    st.subheader("District-level model metrics")
    metrics_table = build_district_model_metrics_table(trained)
    if metrics_table.empty:
        st.info("No district-level model metrics are available yet for this selection.")
    else:
        st.caption("Comparison of Persistence, Random Forest, and XGBoost for the current district split.")
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    st.divider()

    map_col, heatmap_col = st.columns(2)
    with map_col:
        st.subheader("Location map")
        if location_context is None:
            st.info("No coordinates were found for this district, so the map panel falls back to the available district-level context.")
        else:
            map_fig = build_location_map(location_context, featured)
            if map_fig is not None:
                _plotly_chart(map_fig, key="district_home_map")
            else:
                st.info("A map could not be generated for the current selection.")

    with heatmap_col:
        st.subheader("Groundwater heatmap")
        heatmap_fig = build_daily_groundwater_heatmap(featured)
        if heatmap_fig is not None:
            _plotly_chart(heatmap_fig, key="district_home_heatmap")
        else:
            st.info("A groundwater heatmap is not available for this district yet.")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Actual vs forecasted level")
        forecast_fig = build_forecast_figure(featured, forecast_df)
        _plotly_chart(forecast_fig, key="district_forecast_chart")
    with chart_col2:
        st.subheader("Anomaly detection")
        anomaly_fig = build_anomaly_chart(featured)
        _plotly_chart(anomaly_fig, key="district_anomaly_chart")

    st.divider()

    local_reference = load_local_district_reference()
    district_lists_col, district_low_col = st.columns(2)
    with district_lists_col:
        st.subheader("Top 10 anomalous districts")
        if local_reference.empty:
            st.info("Local district reference data is not available yet.")
        else:
            ranked = local_reference[["state", "district", "IF__anomaly_density", "IF__anomaly_count", "gwl_min"]].copy()
            ranked = ranked.dropna(subset=["IF__anomaly_density"])
            if ranked.empty:
                st.info("No district anomaly data is available in the local reference set.")
            else:
                ranked["anomaly_density_pct"] = ranked["IF__anomaly_density"] * 100
                ranked = ranked.sort_values(["anomaly_density_pct", "IF__anomaly_count"], ascending=[False, False]).head(10)
                st.dataframe(ranked[["state", "district", "anomaly_density_pct", "IF__anomaly_count", "gwl_min"]], use_container_width=True, hide_index=True)

    with district_low_col:
        st.subheader("Top 10 lowest groundwater districts")
        if local_reference.empty:
            st.info("Local district reference data is not available yet.")
        else:
            lowest = local_reference[["state", "district", "gwl_min", "gwl_max", "IF__anomaly_density"]].copy()
            lowest = lowest.dropna(subset=["gwl_min"])
            if lowest.empty:
                st.info("No groundwater range data is available in the local reference set.")
            else:
                lowest = lowest.sort_values(["gwl_min", "IF__anomaly_density"], ascending=[True, False]).head(10)
                st.dataframe(lowest[["state", "district", "gwl_min", "gwl_max", "IF__anomaly_density"]], use_container_width=True, hide_index=True)


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

    # Ensure Streamlit metric card labels and values are readable (force black text)
    st.markdown(
        """
        <style>
            [data-testid="stMetricLabel"] {
                color: #000000 !important;
            }
            [data-testid="stMetricValue"] {
                color: #000000 !important;
            }
            [data-testid="stMetric"] {
                color: #000000 !important;
            }
        </style>
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
        height=420,
    )
    metric_fig = apply_research_style(metric_fig, title="Model error metrics", height=420, x_title="Model", y_title="Score")
    _plotly_chart(metric_fig, key="research_metrics_bars")


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

    forecast_fig = build_forecast_figure(trained["featured"], forecast_df)
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
    heat_fig = apply_research_style(heat_fig, title="Regime confusion matrix", height=420, x_title="Predicted regime", y_title="Actual regime")
    _plotly_chart(heat_fig, key="research_regime_confusion_matrix")

    st.dataframe(matrix_df, use_container_width=True)

    regime_accuracy = trained["metrics"]["Regime Acc"].rename("Regime Acc")
    st.write("Regime accuracy by model")
    st.dataframe(regime_accuracy.to_frame(), use_container_width=True)

    coverage = trained["metrics"].loc[:, ["Regime Coverage", "Regime Majority Share"]].head(1)
    st.write("Regime coverage diagnostics")
    st.dataframe(coverage.style.format("{:.3f}"), use_container_width=True)
    st.caption(trained.get("regime_eval_note", ""))


def _load_shap_importance(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "feature"})
    if "feature" not in df.columns and df.shape[1] >= 2:
        df.columns = ["feature", "mean_abs_shap"] + list(df.columns[2:])
    return df[["feature", "mean_abs_shap"]].dropna(subset=["feature"]).head(30)


def render_state_level_dashboard():
    output_dir = os.path.join(parent_dir, "aggregation_outputs")
    state_file = os.path.join(output_dir, "state_level_aggregation.csv")
    anomaly_file = os.path.join(output_dir, "anomaly_density_by_state.csv")
    shap_rf_file = os.path.join(output_dir, "shap_national_Random_Forest.csv")
    shap_xgb_file = os.path.join(output_dir, "shap_national_XGBoost.csv")

    if not os.path.exists(state_file):
        st.warning("State-level aggregation file not found. Run the aggregation script first.")
        return

    df_state = pd.read_csv(state_file)
    if df_state.empty:
        st.info("State-level aggregation file is present but contains no records.")
    else:
        st.subheader("State-level model metrics")
        metric_cols = [
            "state",
            "Persistence__RMSE_mean",
            "Random Forest__RMSE_mean",
            "XGBoost__RMSE_mean",
            "Persistence__MAE_mean",
            "Random Forest__MAE_mean",
            "XGBoost__MAE_mean",
            "Persistence__MAPE_mean",
            "Random Forest__MAPE_mean",
            "XGBoost__MAPE_mean",
            "Persistence__R2_mean",
            "Random Forest__R2_mean",
            "XGBoost__R2_mean",
            "Persistence__NRMSE_mean",
            "Random Forest__NRMSE_mean",
            "XGBoost__NRMSE_mean",
        ]
        available_cols = [col for col in metric_cols if col in df_state.columns]
        st.dataframe(df_state[available_cols].sort_values("state").reset_index(drop=True), use_container_width=True)

        forecast_metric_cols = [
            "Persistence__RMSE_mean",
            "Random Forest__RMSE_mean",
            "XGBoost__RMSE_mean",
        ]
        forecast_chart_cols = [col for col in forecast_metric_cols if col in df_state.columns]
        if forecast_chart_cols:
            st.divider()
            st.subheader("Forecast error comparison across states")
            forecast_df = df_state[["state"] + forecast_chart_cols].copy()
            if "XGBoost__RMSE_mean" in forecast_df.columns:
                forecast_df = forecast_df.sort_values("XGBoost__RMSE_mean", ascending=True)
            else:
                forecast_df = forecast_df.sort_values(forecast_chart_cols[0], ascending=True)

            forecast_fig = go.Figure()
            if "Persistence__RMSE_mean" in forecast_df.columns:
                forecast_fig.add_trace(
                    go.Bar(
                        x=forecast_df["state"],
                        y=forecast_df["Persistence__RMSE_mean"],
                        name="Persistence",
                        marker=dict(color="#6baed6"),
                    )
                )
            if "Random Forest__RMSE_mean" in forecast_df.columns:
                forecast_fig.add_trace(
                    go.Bar(
                        x=forecast_df["state"],
                        y=forecast_df["Random Forest__RMSE_mean"],
                        name="Random Forest",
                        marker=dict(color="#74c476"),
                    )
                )
            if "XGBoost__RMSE_mean" in forecast_df.columns:
                forecast_fig.add_trace(
                    go.Bar(
                        x=forecast_df["state"],
                        y=forecast_df["XGBoost__RMSE_mean"],
                        name="XGBoost",
                        marker=dict(color="#fd8d3c"),
                    )
                )
            forecast_fig.update_layout(barmode="group", height=620)
            forecast_fig.update_xaxes(tickangle=-45)
            forecast_fig = apply_research_style(
                forecast_fig,
                title="State-level model RMSE comparison",
                height=620,
                x_title="State",
                y_title="RMSE",
                showlegend=True,
            )
            _plotly_chart(forecast_fig, key="state_level_rmse_comparison")

    if os.path.exists(anomaly_file):
        df_anom = pd.read_csv(anomaly_file)
        if not df_anom.empty:
            st.divider()
            st.subheader("State anomaly density")
            if "anomaly_mean" in df_anom.columns:
                df_anom = df_anom.rename(columns={"anomaly_mean": "anomaly_density_mean"})
            st.dataframe(df_anom.sort_values("state").reset_index(drop=True), use_container_width=True)

            if "anomaly_density_mean" in df_anom.columns:
                anom_fig = go.Figure(
                    go.Bar(
                        x=df_anom.sort_values("anomaly_density_mean", ascending=False)["state"],
                        y=df_anom.sort_values("anomaly_density_mean", ascending=False)["anomaly_density_mean"],
                        marker=dict(
                            color=df_anom.sort_values("anomaly_density_mean", ascending=False)["anomaly_density_mean"],
                            colorscale="YlOrRd",
                        ),
                    )
                )
                anom_fig.update_xaxes(tickangle=-45)
                anom_fig = apply_research_style(
                    anom_fig,
                    title="State-wise anomaly density",
                    height=520,
                    x_title="State",
                    y_title="Mean anomaly density",
                    showlegend=False,
                )
                _plotly_chart(anom_fig, key="state_level_anomaly_density")
        else:
            st.info("State anomaly density file is present but empty.")
    else:
        st.warning("State anomaly density file not found in aggregation outputs.")

    shap_rf = _load_shap_importance(shap_rf_file)
    shap_xgb = _load_shap_importance(shap_xgb_file)
    if shap_rf is None and shap_xgb is None:
        st.warning("No aggregated SHAP importance files were found for state-level visualization.")
        return

    st.divider()
    st.subheader("Aggregated SHAP importances")
    cols = st.columns(2)
    if shap_rf is not None:
        with cols[0]:
            rf_fig = go.Figure(go.Bar(x=shap_rf["mean_abs_shap"], y=shap_rf["feature"], orientation="h", marker=dict(color="#2b8cbe")))
            rf_fig = apply_research_style(rf_fig, title="Random Forest SHAP importance", height=520, x_title="Mean |SHAP|", y_title="Feature", showlegend=False)
            _plotly_chart(rf_fig, key="state_level_shap_rf")
    if shap_xgb is not None:
        with cols[1]:
            xgb_fig = go.Figure(go.Bar(x=shap_xgb["mean_abs_shap"], y=shap_xgb["feature"], orientation="h", marker=dict(color="#d95f02")))
            xgb_fig = apply_research_style(xgb_fig, title="XGBoost SHAP importance", height=520, x_title="Mean |SHAP|", y_title="Feature", showlegend=False)
            _plotly_chart(xgb_fig, key="state_level_shap_xgb")


def render_national_aggregation_dashboard():
    """Render the national-level aggregation metrics dashboard from cached aggregation files."""
    output_dir = os.path.join(parent_dir, "aggregation_outputs")

    if not os.path.exists(output_dir):
        st.warning("Aggregation outputs directory not found. Run the aggregation script first.")
        return

    national_file = os.path.join(output_dir, "national_level_aggregation.csv")
    anomaly_file = os.path.join(output_dir, "anomaly_density_by_state.csv")
    shap_rf_file = os.path.join(output_dir, "shap_national_Random_Forest.csv")
    shap_xgb_file = os.path.join(output_dir, "shap_national_XGBoost.csv")

    if not os.path.exists(national_file):
        st.warning("National-level aggregation file not found.")
        return

    df_national = pd.read_csv(national_file)
    if df_national.empty:
        st.info("National-level aggregation file is present but contains no records.")
    else:
        st.subheader("National model metrics")
        table_columns = [
            "Model",
            "n_districts",
            "total_obs",
            "RMSE_mean",
            "MAE_mean",
            "MAPE_mean",
            "R2_mean",
            "mNSE_mean",
            "KGE_mean",
            "NRMSE_mean",
        ]
        available_cols = [col for col in table_columns if col in df_national.columns]
        st.dataframe(df_national[available_cols].reset_index(drop=True), use_container_width=True)

        st.divider()
        st.subheader("National forecast comparison")
        forecast_fig = build_national_metric_figure(df_national)
        _plotly_chart(forecast_fig, key="national_metric_comparison")

    if os.path.exists(anomaly_file):
        df_anom = pd.read_csv(anomaly_file)
        if df_anom.empty:
            st.info("Anomaly density file is present but empty.")
        else:
            st.divider()
            st.subheader("India anomaly density")
            if "anomaly_mean" in df_anom.columns:
                df_anom = df_anom.rename(columns={"anomaly_mean": "anomaly_density_mean"})
            st.dataframe(df_anom.sort_values("state").reset_index(drop=True), use_container_width=True)

            st.divider()
            st.subheader("India anomaly density map")
            anomaly_map = build_national_anomaly_map(df_anom)
            _plotly_chart(anomaly_map, key="national_anomaly_map")

            st.divider()
            st.subheader("Top anomalous states")
            anomaly_bar = build_national_anomaly_bar(df_anom)
            _plotly_chart(anomaly_bar, key="national_anomaly_bar")
    else:
        st.warning("Anomaly density file not found in aggregation outputs.")

    shap_rf = _load_shap_importance(shap_rf_file)
    shap_xgb = _load_shap_importance(shap_xgb_file)
    if shap_rf is None and shap_xgb is None:
        st.warning("No aggregated SHAP importance files were found for national-level visualization.")
        return

    st.divider()
    st.subheader("National aggregated SHAP importances")
    cols = st.columns(2)
    if shap_rf is not None:
        with cols[0]:
            rf_fig = go.Figure(go.Bar(x=shap_rf["mean_abs_shap"], y=shap_rf["feature"], orientation="h", marker=dict(color="#0f4c81")))
            rf_fig = apply_research_style(rf_fig, title="Random Forest SHAP importance", height=520, x_title="Mean |SHAP|", y_title="Feature", showlegend=False)
            _plotly_chart(rf_fig, key="national_level_shap_rf")
    if shap_xgb is not None:
        with cols[1]:
            xgb_fig = go.Figure(go.Bar(x=shap_xgb["mean_abs_shap"], y=shap_xgb["feature"], orientation="h", marker=dict(color="#d95f02")))
            xgb_fig = apply_research_style(xgb_fig, title="XGBoost SHAP importance", height=520, x_title="Mean |SHAP|", y_title="Feature", showlegend=False)
            _plotly_chart(xgb_fig, key="national_level_shap_xgb")


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
            color: #000000 !important;
        }
        [data-testid="stMetric"] span,
        [data-testid="stMetric"] div,
        [data-testid="stMetric"] p {
            color: #000000 !important;
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
            district_key = f"district_{state}"
            if st.session_state.get(district_key) not in district_choices:
                st.session_state[district_key] = default_district
            district = st.selectbox("District", district_choices, key=district_key)
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
            raw_df = None
            cached_df = load_cached_data()
            if cached_df is not None and not cached_df.empty:
                raw_df = cached_df.copy()
                st.caption("Using the local cached data for the district-level view.")
            else:
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
                    raw_df = None

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

        home_tab, research_tab, national_tab = st.tabs(["District Level", "State Level", "National Level"])

        with home_tab:
            st.markdown("### District-level dashboard")
            sample, shap_values = shap_summary(trained["xgb"], trained["featured"], feature_cols)
            render_district_level_home_tab(state, district, trained["featured"], forecast_df, trained, location_context, shap_values, feature_cols)

            st.divider()
            render_pattern_panel(trained, feature_cols)

            st.divider()
            render_shap_explanation_panel(feature_cols, shap_values)

        with research_tab:
            render_state_level_dashboard()

        with national_tab:
            render_national_aggregation_dashboard()

        st.success("Dashboard ready: map, heatmaps, explainability, and research validation are available in a simplified layout.")

    else:
        st.info("Choose your parameters in the sidebar and run the analysis.")


if __name__ == "__main__":
    main()