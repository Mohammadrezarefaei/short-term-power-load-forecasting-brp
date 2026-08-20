"""Streamlit Web App: Short-Term Power Load Forecasting & BRP Settlement."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.forecasting_engine import LoadForecastingEngine

st.set_page_config(
    page_title="Short-Term Load Forecaster & BRP Settlement",
    page_icon="📈",
    layout="wide"
)

st.title("📈⚡ Short-Term Power Load Forecaster & BRP Imbalance Engine")
st.markdown("Machine learning time-series regression (**Gradient Boosting**) with weather feature engineering and **German reBAP imbalance penalty** valuation.")

@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    hours = 8760 * 2
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
    day_of_year = timestamps.dayofyear.to_numpy()
    hour_of_day = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)

    temp = 10.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 20) / 365) + 4.5 * np.sin((hour_of_day - 8) * np.pi / 12) + np.random.normal(0, 2.0, hours)
    heat = np.maximum(0, 15.0 - temp) * 850.0
    cool = np.maximum(0, temp - 22.0) * 1100.0

    load = 48000.0 + 14000.0 * np.sin((hour_of_day - 6) * np.pi / 12) - (is_weekend * 11000.0) + heat + cool + np.random.normal(0, 800.0, hours)
    return pd.DataFrame({
        "timestamp": timestamps,
        "actual_load_mw": np.clip(load, 32000.0, 78000.0),
        "temperature_c": temp
    })

# Sidebar Controls
st.sidebar.header("⚙️ Model Configuration")
train_split_pct = st.sidebar.slider("Training Set Size (Months)", 6, 18, 12, 1)
sample_week_idx = st.sidebar.slider("Inspection Week (Hour Offset)", 0, 7000, 2000, 168)

df_raw = generate_synthetic_data()
engine = LoadForecastingEngine()
df_features = engine.engineer_features(df_raw)

train_hours = train_split_pct * 730
df_test, metrics = engine.train_and_evaluate(df_features, train_split_hours=train_hours)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Out-of-Sample 7-Day Forecast Zoom")
    df_zoom = df_test.iloc[sample_week_idx : sample_week_idx + 168]
    
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df_zoom["timestamp"], df_zoom["actual_load_mw"] / 1000.0, label="Actual Grid Demand (GW)", color="#1E293B", lw=2.0)
    ax.plot(df_zoom["timestamp"], df_zoom["predicted_load_mw"] / 1000.0, label="Gradient Boosted Forecast (GW)", color="#2563EB", lw=1.8, linestyle="--")
    ax.fill_between(
        df_zoom["timestamp"],
        df_zoom["actual_load_mw"] / 1000.0,
        df_zoom["predicted_load_mw"] / 1000.0,
        color="#EF4444",
        alpha=0.25,
        label="Forecast Deviation"
    )
    ax.set_ylabel("Grid Demand [GW]", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Forecast & BRP Commercial Metrics")
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['mape_pct']:.2f} %")
    st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse_mw']:,.1f} MW")
    st.metric("Mean Absolute Error (MAE)", f"{metrics['mean_abs_error_mw']:,.1f} MW")
    st.metric("Simulated BRP Imbalance Penalty", f"€{metrics['total_imbalance_penalty_eur']:,.2f}")

st.markdown("---")
st.caption("Demonstrates the direct link between short-term machine learning load forecasting accuracy and portfolio imbalance settlement exposure in European power markets.")
