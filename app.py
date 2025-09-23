import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="JalDarpan", page_icon="🌊", layout="wide")
st.title("🌊 JalDarpan: Jal ka Sach, Har Pal")

# -----------------------
# Fetch Groundwater Data
# -----------------------
def fetch_groundwater_data(state, district, agency, startdate, enddate, size=500):
    url = "https://indiawris.gov.in/Dataset/Ground%20Water%20Level"
    payload = {
        "stateName": state,
        "districtName": district,
        "agencyName": agency,
        "startdate": startdate,
        "enddate": enddate,
        "download": "false",
        "page": 0,
        "size": size
    }

    try:
        st.info("Fetching live data from IndiaWRIS API...")
        response = requests.post(url, data=payload, timeout=10) 
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            raise ValueError("Empty API response")
        df = pd.DataFrame(data)
        st.success(f"Fetched {len(df)} records from API.")
        return df
    except Exception as e:
        st.warning(f"Could not fetch live data. Using cached data instead. ({e})")
        # Load cached CSV
        try:
            df = pd.read_csv("cached_groundwater_data.csv")
            st.success(f"Loaded {len(df)} records from cached data.")
            return df
        except Exception as e2:
            st.error(f"Failed to load cached data: {e2}")
            return None

# -----------------------
# Preprocess Data
# -----------------------
def preprocess_data(df):
    df = df[['dataTime','dataValue']].dropna()
    df['date'] = pd.to_datetime(df['dataTime'])
    df['groundwater_level'] = pd.to_numeric(df['dataValue'], errors='coerce')
    df = df.dropna(subset=['groundwater_level'])
    df = df.sort_values('date').set_index('date')

    # Features
    for lag in range(1, 6):
        df[f'lag{lag}'] = df['groundwater_level'].shift(lag)
    df['rolling7'] = df['groundwater_level'].rolling(7, min_periods=1).mean()
    df['rolling30'] = df['groundwater_level'].rolling(30, min_periods=1).mean()
    df['diff1'] = df['groundwater_level'].diff(1)
    df['diff2'] = df['groundwater_level'].diff(2)
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df = df.dropna()
    return df

# -----------------------
# Load Pretrained Models
# -----------------------
def load_models():
    try:
        rf = joblib.load("rf_model.pkl")
        xgb = joblib.load("xgb_model.pkl")
        lstm = joblib.load("lstm_model.pkl")
        return rf, xgb, lstm
    except Exception as e:
        st.warning(f"Could not load pretrained models: {e}")
        return None, None, None

# -----------------------
# Sidebar Inputs
# -----------------------
state = st.sidebar.text_input("State Name", "Odisha")
district = st.sidebar.text_input("District Name", "Baleshwar")
agency = st.sidebar.text_input("Agency Name", "CGWB")
startdate = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
enddate = st.sidebar.date_input("End Date", datetime.now())
size = st.sidebar.number_input("Records (max 1000)", min_value=1, max_value=1000, value=500)

if st.sidebar.button("Fetch Data"):
    df = fetch_groundwater_data(state, district, agency,
                                startdate.strftime("%Y-%m-%d"),
                                enddate.strftime("%Y-%m-%d"),
                                size)
    if df is not None and not df.empty:
        st.subheader("📊 Raw Data")
        st.write(df)

        df_clean = preprocess_data(df)
        st.subheader("🧹 Cleaned Data")
        st.write(df_clean)

        st.subheader("📈 Groundwater Level Over Time")
        st.line_chart(df_clean['groundwater_level'])

        # Load pretrained models
        rf, xgb, lstm = load_models()
        if rf and xgb:
            st.success("✅ Pretrained ML models loaded successfully. Ready for prediction!")
            features_used = ['lag1','lag2','lag3','lag4','lag5','rolling7','rolling30',
                             'diff1','diff2','month','weekofyear']
            
            # Predictions
            y_test = df_clean['groundwater_level']
            pred_rf = rf.predict(df_clean[features_used])
            pred_xgb = xgb.predict(df_clean[features_used])

            # Plot predictions
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(y_test.index, y_test.values, label='Actual', color='black')
            ax.plot(y_test.index, pred_rf, label='Random Forest', alpha=0.7)
            ax.plot(y_test.index, pred_xgb, label='XGBoost', alpha=0.7)
            ax.set_title("Groundwater Level Prediction Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("Groundwater Level (m)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("ML models not loaded. Only raw and cleaned data displayed.")
