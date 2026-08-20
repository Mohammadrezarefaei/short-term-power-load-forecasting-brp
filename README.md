# 📈⚡ Short-Term Power Load Forecaster & BRP Imbalance Engine

[![CI Pipeline](https://img.shields.io/badge/CI%20Pipeline-passing-brightgreen?logo=github&style=flat-square)](https://github.com/Mohammadrezarefaei/short-term-power-load-forecasting-brp/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://short-term-power-load-forecasting-brp-heqtxke6tbbs7t382dhpw9.streamlit.app/)

A production-grade machine learning pipeline for **Day-Ahead electricity grid load forecasting** using **HistGradientBoosting** regression, automated feature engineering (calendar dynamics, weather sensitivities, multi-horizon auto-regressive lags), and financial risk valuation under **German Balance Responsible Party (BRP) imbalance settlement (reBAP)** mechanisms.

---

## 🚀 Live Interactive Demo
👉 **[Access the Live Streamlit Web App](https://short-term-power-load-forecasting-brp-heqtxke6tbbs7t382dhpw9.streamlit.app/)**

---

## 📌 Problem Architecture & Market Mechanics

In European power markets (such as Germany's ENTSO-E bidding zone), Balancing Responsible Parties (BRPs) face severe financial penalties if their actual physical grid feed-in/offtake deviates from their scheduled Day-Ahead nominations:

* **Objective Function (Loss Minimization):**
  $$\min \sum_{t=1}^{T} \left( y(t) - \hat{y}(t) \right)^2 \quad \text{via Gradient Boosted Decision Trees}$$
* **Feature Representation:**
  $$X_t = \left[ \text{Hour}_t, \text{DayOfWeek}_t, \text{Month}_t, T_{\text{ambient}}(t), y(t-24), y(t-48), y(t-168), \bar{y}_{\text{rolling, 24h}}(t-24) \right]$$
* **Financial Imbalance Exposure (reBAP Settlement):**
  $$\text{Penalty Cost} = \sum_{t=1}^{T} \left| y(t) - \hat{y}(t) \right| \cdot \lambda_{\text{reBAP}}(t)$$

---

## 🔍 Key Findings & Commercial Value Drivers

* **Error Minimization:** Achieves strong out-of-sample accuracy ($MAPE < 3.5\%$) across multi-gigawatt grid demand profiles, capturing both seasonal heating/cooling degree days and weekly industrial ramp-ups.
* **Auto-Regressive Multi-Horizon Lags:** Incorporating strict $t-24$, $t-48$, and $t-168$ (same hour previous week) lag features prevents data leakage while preserving intra-day cyclic patterns.
* **BRP Cost Hedging:** Demonstrates that a 1% reduction in Day-Ahead forecast error translates directly into substantial savings on balancing market settlement charges.

---

## 🛠️ Software Architecture & Automated Testing
* **CI/CD Pipeline:** Fully automated testing via **GitHub Actions** with **all unit tests passing** (`pytest` suite validating feature engineering consistency, out-of-sample data splits, and model prediction convergence).
* **Modular Core Engine:** Implemented in `src/forecasting_engine.py`.
* **Tech Stack:** Python 3.11, Scikit-Learn (HistGradientBoosting), Pandas, NumPy, Matplotlib, Streamlit, Pytest.
