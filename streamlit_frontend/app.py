import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap
import os

def fetch_groundwater_data(state, district, agency, startdate, enddate, size=500):
    # Call Flask proxy endpoint
    flask_url = f"https://jaldarpan-flask.azurewebsites.net/groundwater?state={state}&district={district}&agency={agency}&startdate={startdate}&enddate={enddate}&size={size}"
    try:
        st.info("Fetching live data from Flask proxy...")
        response = requests.get(flask_url, timeout=20)
        response.raise_for_status()
        json_data = response.json()
        if json_data.get("status") != "success":
            raise ValueError(json_data.get("message", "Flask error"))
        data = json_data.get("data", [])
        if not data:
            st.warning("Flask returned empty data. Using cached data instead.")
            raise ValueError("Empty API response")
        df = pd.DataFrame(data)
        st.success(f"Fetched {len(df)} records from Flask proxy.")
        st.write("Columns from API:", df.columns.tolist())
        return df
    except Exception as e:
        st.warning(f"Could not fetch from Flask. Using cached data instead. ({e})")
        try:
            cached_file = "cached_groundwater_data.csv"
            if os.path.exists(cached_file):
                df = pd.read_csv(cached_file)
                st.success(f"Loaded {len(df)} records from cached data.")
                return df
            else:
                st.error("Cached file not found.")
                return None
        except Exception as e2:
            st.error(f"Failed to load cached data: {e2}")
            return None

def preprocess_data(df):
    df = df[['dataTime', 'dataValue']].copy()
    df = df.dropna(subset=['dataTime', 'dataValue'])
    
    df['date'] = pd.to_datetime(df['dataTime'])
    df['groundwater_level'] = pd.to_numeric(df['dataValue'], errors='coerce')
    df = df.dropna(subset=['groundwater_level'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    for lag in range(1, 6):
        df[f'lag{lag}'] = df['groundwater_level'].shift(lag)

    df['rolling7'] = df['groundwater_level'].rolling(7, min_periods=1).mean()
    df['rolling30'] = df['groundwater_level'].rolling(30, min_periods=1).mean()
    df['rolling7_std'] = df['groundwater_level'].rolling(7, min_periods=1).std()
    df['rolling30_std'] = df['groundwater_level'].rolling(30, min_periods=1).std()

    df['diff1'] = df['groundwater_level'].diff(1)
    df['diff2'] = df['groundwater_level'].diff(2)

    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    df = df.dropna()
    
    return df

def train_and_evaluate_models(df):
    feature_cols = ['lag1','lag2','lag3','lag4','lag5',
                    'rolling7','rolling30','rolling7_std','rolling30_std',
                    'diff1','diff2','month','weekofyear']
    X = df[feature_cols]
    y = df['groundwater_level']

    if len(X) < 10:
        st.warning("Not enough data to train models. Increase date range or check API response.")
        return None, None, None, None, None, None, None

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    results = {}

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
        "MAE": mean_absolute_error(y_test, pred_rf),
        "R2": r2_score(y_test, pred_rf),
        "Parameters": rf.get_params()
    }

    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_xgb)),
        "MAE": mean_absolute_error(y_test, pred_xgb),
        "R2": r2_score(y_test, pred_xgb),
        "Parameters": xgb.get_params()
    }

    X_lstm = np.array(X).reshape((len(X), X.shape[1], 1))
    y_lstm = np.array(y)
    X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
    y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

    model_lstm = Sequential()
    model_lstm.add(LSTM(100, activation='relu', input_shape=(X_train_lstm.shape[1],1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=25, verbose=0)
    pred_lstm = model_lstm.predict(X_test_lstm).flatten()

    results['LSTM'] = {
        "RMSE": np.sqrt(mean_squared_error(y_test_lstm, pred_lstm)),
        "MAE": mean_absolute_error(y_test_lstm, pred_lstm),
        "R2": r2_score(y_test_lstm, pred_lstm),
        "Parameters": {
            "units": 100,
            "activation": "relu",
            "epochs": 25
        }
    }

    return results, y_test, pred_rf, pred_xgb, pred_lstm, rf, xgb

def explain_models_dynamic(results):
    for model_name, metrics in results.items():
        st.markdown(f"#### {model_name} Model")
        st.write(f"**Model Parameters:** {metrics.get('Parameters', 'N/A')}")
        
        rmse = metrics.get("RMSE", None)
        mae = metrics.get("MAE", None)
        r2 = metrics.get("R2", None)
        
        if rmse is not None:
            st.write(f"- **RMSE (Root Mean Squared Error):** {rmse:.3f} → Avg prediction error in meters.")
        if mae is not None:
            st.write(f"- **MAE (Mean Absolute Error):** {mae:.3f} → Avg absolute difference between predicted and actual values.")
        if r2 is not None:
            st.write(f"- **R² (Coefficient of Determination):** {r2:.3f} → How well the model explains variance (closer to 1 is better).")
        
        if r2 is not None:
            if r2 < 0.5:
                st.warning("⚠️ Model may underfit. Consider more features or tuning.")
            elif r2 > 0.8:
                st.success("✅ Model fits well and captures trends effectively.")
            else:
                st.info("ℹ️ Model performance is moderate; may require improvement.")

def explainable_ai(model, X_train, model_name):
    st.subheader(f"🔍 Explainable AI for {model_name}")
    
    if model_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_train, check_additivity=False)

        st.write("**Feature Importance (SHAP Summary Plot)**")
        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)

        st.write("**Detailed SHAP Plot**")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig2)
    else:
        st.info("Explainability for LSTM is more complex (sequence-based). Consider Integrated Gradients or attention mechanisms.")

st.set_page_config(page_title="JalDarpan", page_icon="🌊", layout="wide")
st.title("🌊 JalDarpan: Jal ka Sach, Har Pal")

# Sidebar inputs
state = st.sidebar.text_input("State Name", "Odisha")
district = st.sidebar.text_input("District Name", "Baleshwar")
agency = st.sidebar.text_input("Agency Name", "CGWB")
startdate = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
enddate = st.sidebar.date_input("End Date", datetime.now())
size = st.sidebar.number_input("Records (max 1000)", min_value=1, max_value=1000, value=500)

if st.sidebar.button("Fetch Data"):
    df = fetch_groundwater_data(
        state, district, agency,
        startdate.strftime("%Y-%m-%d"),
        enddate.strftime("%Y-%m-%d"),
        size
    )

    if df is not None and not df.empty:
        st.subheader("📊 Raw Data")
        st.write(df)

        df_clean = preprocess_data(df)
        if df_clean is not None and not df_clean.empty:
            st.subheader("🧹 Cleaned & Processed Data")
            st.write(df_clean)

            st.subheader("📈 Groundwater Level Over Time")
            st.line_chart(df_clean['groundwater_level'])

            st.subheader("⚡ Model Training & Evaluation")
            results, y_test, pred_rf, pred_xgb, pred_lstm, rf, xgb = train_and_evaluate_models(df_clean)

            if results is not None:
                results_df = pd.DataFrame({k:{k2:v for k2,v in val.items() if k2 != 'Parameters'} for k,val in results.items()}).T
                st.subheader("📊 Model Comparison Table")
                st.write(results_df)

                # Plot predictions
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(y_test.index, y_test.values, label='Actual', color='black')
                ax.plot(y_test.index, pred_rf, label='Random Forest', alpha=0.7)
                ax.plot(y_test.index, pred_xgb, label='XGBoost', alpha=0.7)
                ax.plot(y_test.index, pred_lstm, label='LSTM', alpha=0.7)
                ax.set_title("Groundwater Level Prediction Comparison")
                ax.set_xlabel("Date")
                ax.set_ylabel("Groundwater Level (m)")
                ax.legend()
                st.pyplot(fig)

                # Dynamic explanations
                st.subheader("📘 Model Interpretability & Dynamic Explanations")
                explain_models_dynamic(results)
                features_used = ['lag1','lag2','lag3','lag4','lag5',
                 'rolling7','rolling30','rolling7_std','rolling30_std',
                 'diff1','diff2','month','weekofyear']

                explainable_ai(rf, df_clean[features_used], "Random Forest")
                explainable_ai(xgb, df_clean[features_used], "XGBoost")

        else:
            st.warning("No valid data after preprocessing. Try a larger date range.")