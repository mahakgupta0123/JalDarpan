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


def _safe_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


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
            "Persistence": evaluate(y_test.values, persistence_pred),
            "Random Forest": evaluate(y_test.values, rf_pred),
            "XGBoost": evaluate(y_test.values, xgb_pred),
        }
    ).T

    metrics["Regime Acc"] = [persistence_regime_accuracy, rf_regime_accuracy, xgb_regime_accuracy]
    metrics["Regime Acc"] = metrics["Regime Acc"].astype(float)

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
    anomaly_rate = float(featured["anomaly_flag"].mean() * 100)
    trend_window = featured["groundwater_level"].tail(min(14, len(featured)))
    trend_delta = float(trend_window.iloc[-1] - trend_window.iloc[0]) if len(trend_window) > 1 else 0.0
    next_forecast = float(forecast_df["forecast"].iloc[0]) if forecast_df is not None and not forecast_df.empty else np.nan

    cols = st.columns(4)
    cols[0].metric("Latest level", f"{last_value:.2f}")
    cols[1].metric("7-day forecast", f"{next_forecast:.2f}" if np.isfinite(next_forecast) else "N/A")
    cols[2].metric("Anomaly rate", f"{anomaly_rate:.1f}%")
    cols[3].metric("14-day trend", f"{trend_delta:+.2f}")


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
        "Top SHAP drivers surfaced as text for paper-ready explanation.",
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
        readable_names = []
        for feature_name in top_features.index:
            if feature_name.startswith("lag_"):
                readable_names.append(f"recent history ({feature_name.replace('_', ' ')})")
            elif feature_name.startswith("rolling_"):
                readable_names.append(f"short-term average or variability ({feature_name.replace('_', ' ')})")
            elif feature_name in {"month", "dayofyear", "weekday", "is_monsoon"}:
                readable_names.append(f"seasonality and calendar effects ({feature_name.replace('_', ' ')})")
            else:
                readable_names.append(feature_name.replace("_", " "))

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


def render_confusion_matrix_panel(trained):
    st.subheader("Regime-based confusion matrix")
    st.caption("Because the target is continuous, the dashboard converts groundwater levels into Low / Moderate / High regimes using train-set quantiles, then compares predicted regimes against actual regimes.")

    matrix_choice = st.selectbox("Choose model", ["Persistence", "Random Forest", "XGBoost"], index=2)
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
    heat_fig.update_layout(height=420, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(heat_fig, use_container_width=True)

    st.dataframe(matrix_df, use_container_width=True)

    regime_accuracy = trained["metrics"]["Regime Acc"].rename("Regime Acc")
    st.write("Regime accuracy by model")
    st.dataframe(regime_accuracy.to_frame(), use_container_width=True)


def render_plain_language_panels(trained, forecast_df, feature_cols, shap_values):
    groundwater_story = describe_groundwater_status(trained["featured"], forecast_df)
    xai_story = describe_xai_meaning(feature_cols, shap_values)
    next_forecast_display = f"{groundwater_story['next_forecast']:.2f}" if np.isfinite(groundwater_story["next_forecast"]) else "N/A"

    st.subheader("Plain-language groundwater status")
    st.markdown(
        f"""
        <div style="padding:1rem 1.1rem;border-radius:16px;background:#f7fbff;border:1px solid #d7e7f5;">
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
        <div style="padding:1rem 1.1rem;border-radius:16px;background:#fffaf2;border:1px solid #f0dcc0;">
            <p style="margin:0 0 0.35rem 0;font-weight:700;font-size:1.05rem;">What SHAP is saying</p>
            <p style="margin:0;">{xai_story['summary_text']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(xai_story["plain_language"])
    st.dataframe(xai_story["top_features"].rename("mean_abs_shap").to_frame(), use_container_width=True)


def main():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0b2b40 0%, #124559 55%, #1f6f78 100%);
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(0,0,0,0.18);
        }
        .subtle { opacity: 0.86; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">JalDarpan Research Dashboard</h1>
            <p class="subtle" style="margin:0.35rem 0 0 0;">Time-series groundwater analytics with forecasting, anomaly detection, and explainability.</p>
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

        forecast_df = forecast_next_steps(
            trained["xgb"],
            featured[["groundwater_level"]],
            feature_cols,
            steps=int(forecast_horizon),
            interval_low=trained["interval_low"],
            interval_high=trained["interval_high"],
        )

        render_kpis(trained["featured"], trained["metrics"], forecast_df)

        overview_tab, forecast_tab, metrics_tab, explain_tab, method_tab, raw_tab = st.tabs(
            ["Overview", "Forecast & Anomalies", "Metrics", "Explainability", "Methodology", "Raw Data"]
        )

        with overview_tab:
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(
                go.Scatter(
                    x=trained["featured"].index,
                    y=trained["featured"]["groundwater_level"],
                    name="Observed",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            anomaly_points = trained["featured"][trained["featured"]["anomaly_flag"] == 1]
            if not anomaly_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points.index,
                        y=anomaly_points["groundwater_level"],
                        name="Anomaly",
                        mode="markers",
                        marker=dict(color="#d62728", size=9, symbol="x"),
                    )
                )
            fig.update_layout(
                height=520,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h"),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Data Snapshot")
            st.dataframe(trained["featured"].tail(20), use_container_width=True)

        with forecast_tab:
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
                height=520,
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True)

        with metrics_tab:
            st.subheader("Model comparison")
            st.dataframe(trained["metrics"].style.format("{:.4f}"), use_container_width=True)
            st.caption("Persistence is included as a simple baseline to make the research evaluation defensible.")

            metric_fig = go.Figure()
            for metric_name in ["RMSE", "MAE", "MAPE"]:
                metric_fig.add_trace(
                    go.Bar(
                        x=trained["metrics"].index,
                        y=trained["metrics"][metric_name],
                        name=metric_name,
                    )
                )
            metric_fig.update_layout(barmode="group", template="plotly_white", height=480)
            st.plotly_chart(metric_fig, use_container_width=True)

            render_confusion_matrix_panel(trained)

        with explain_tab:
            st.subheader("SHAP feature importance for XGBoost")
            sample, shap_values = shap_summary(trained["xgb"], trained["featured"], feature_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            st.pyplot(fig, clear_figure=True)

            st.subheader("Research interpretation")
            top_features = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
            st.write(
                "Top drivers: "
                + ", ".join(top_features.head(5).index.tolist())
            )
            st.dataframe(top_features.rename("mean_abs_shap").to_frame(), use_container_width=True)

            st.info(
                "Interpretation: positive SHAP values push the prediction toward higher groundwater depth, while negative values pull it lower; the ranking shows which engineered time-series signals matter most for the model."
            )

            st.divider()
            render_plain_language_panels(trained, forecast_df, feature_cols, shap_values)

        with method_tab:
            st.subheader("Research methodology")
            render_methodology_cards(trained["featured"], feature_cols, trained)
            st.markdown("### Summary of model logic")
            st.write(
                "The pipeline first cleans and standardizes WRIS observations, then extracts lag and rolling statistics to transform the raw series into a supervised learning table. XGBoost is used as the primary predictor because it handles nonlinear temporal interactions well; Isolation Forest flags unusual readings; and SHAP explains which features drive the forecast."
            )

        with raw_tab:
            st.subheader("Raw input data")
            st.dataframe(raw_df, use_container_width=True)
            st.download_button(
                "Download cleaned features as CSV",
                data=trained["featured"].reset_index().to_csv(index=False).encode("utf-8"),
                file_name="jaldarpan_features.csv",
                mime="text/csv",
            )

        st.success("Research-grade pipeline complete: data ingestion, feature engineering, forecasting, anomaly detection, and explainability.")

    else:
        st.info("Choose your parameters in the sidebar and run the analysis.")


if __name__ == "__main__":
    main()