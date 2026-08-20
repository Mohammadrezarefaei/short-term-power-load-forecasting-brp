"""Automated Pytest Suite for Power Load Forecasting Engine."""

import pytest
import numpy as np
import pandas as pd
from src.forecasting_engine import LoadForecastingEngine


@pytest.fixture
def sample_load_data():
    np.random.seed(42)
    hours = 8760 + 500  # Train + test + lag warm-up
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
    load = 50000.0 + 10000.0 * np.sin(np.arange(hours) * np.pi / 12) + np.random.normal(0, 500, hours)
    temp = 12.0 + 8.0 * np.cos(np.arange(hours) * 2 * np.pi / 8760)
    return pd.DataFrame({
        "timestamp": timestamps,
        "actual_load_mw": load,
        "temperature_c": temp
    })


def test_feature_engineering_integrity(sample_load_data):
    engine = LoadForecastingEngine()
    df_feat = engine.engineer_features(sample_load_data)
    
    for col in engine.feature_cols:
        assert col in df_feat.columns
    assert not df_feat[engine.feature_cols].isnull().any().any()


def test_model_training_and_metrics_validity(sample_load_data):
    engine = LoadForecastingEngine()
    df_feat = engine.engineer_features(sample_load_data)
    df_test, metrics = engine.train_and_evaluate(df_feat, train_split_hours=8000)

    assert "predicted_load_mw" in df_test.columns
    assert metrics["mape_pct"] > 0.0
    assert metrics["mape_pct"] < 25.0  # Must be a reasonable forecast error
    assert metrics["rmse_mw"] > 0.0
    assert metrics["total_imbalance_penalty_eur"] > 0.0
