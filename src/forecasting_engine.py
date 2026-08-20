"""
Short-Term Power Load Forecasting & BRP Imbalance Settlement Engine.
Handles lag feature engineering, gradient boosting regression, and financial error valuation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


class LoadForecastingEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.07,
            max_leaf_nodes=31,
            random_state=self.random_state
        )
        self.feature_cols = [
            "hour", "day_of_week", "is_weekend", "day_of_year", "month",
            "temperature_c", "lag_24h", "lag_48h", "lag_168h", "rolling_mean_24h"
        ]

    def engineer_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Constructs temporal calendar tags and rolling time-series lag features."""
        df = df_raw.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["month"] = df["timestamp"].dt.month

        # Lag & rolling window aggregates
        df["lag_24h"] = df["actual_load_mw"].shift(24)
        df["lag_48h"] = df["actual_load_mw"].shift(48)
        df["lag_168h"] = df["actual_load_mw"].shift(168)
        df["rolling_mean_24h"] = df["actual_load_mw"].shift(24).rolling(window=24).mean()

        return df.dropna().reset_index(drop=True)

    def train_and_evaluate(
        self, df_features: pd.DataFrame, train_split_hours: int = 8760
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Trains the gradient booster and evaluates out-of-sample forecast accuracy."""
        df_train = df_features.iloc[:train_split_hours]
        df_test = df_features.iloc[train_split_hours:].copy().reset_index(drop=True)

        X_train, y_train = df_train[self.feature_cols], df_train["actual_load_mw"]
        X_test, y_test = df_test[self.feature_cols], df_test["actual_load_mw"]

        self.model.fit(X_train, y_train)
        df_test["predicted_load_mw"] = self.model.predict(X_test)
        df_test["forecast_error_mw"] = df_test["actual_load_mw"] - df_test["predicted_load_mw"]
        df_test["abs_error_mw"] = np.abs(df_test["forecast_error_mw"])

        mape = float(mean_absolute_percentage_error(y_test, df_test["predicted_load_mw"]) * 100)
        rmse = float(root_mean_squared_error(y_test, df_test["predicted_load_mw"]))

        # Simulated German BRP Imbalance Settlement (reBAP) Risk
        rebap_prices = np.clip(65.0 + np.random.normal(0, 35.0, len(df_test)), 15.0, 300.0)
        df_test["rebap_price_eur_mwh"] = rebap_prices
        # Financial penalty for 1 GW portfolio deviation
        df_test["imbalance_penalty_eur"] = (df_test["abs_error_mw"] / 1000.0) * rebap_prices * 10.0

        metrics = {
            "mape_pct": round(mape, 2),
            "rmse_mw": round(rmse, 1),
            "mean_abs_error_mw": round(float(df_test["abs_error_mw"].mean()), 1),
            "total_imbalance_penalty_eur": round(float(df_test["imbalance_penalty_eur"].sum()), 2)
        }

        return df_test, metrics
