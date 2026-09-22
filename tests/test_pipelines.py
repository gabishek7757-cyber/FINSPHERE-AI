"""
FinSphere AI — Test Suite
Tests feature engineering, metrics calculations, PSI drift, pipeline predictions, and API endpoints.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.metrics_utils import (
    evaluate_classification_metrics,
    find_optimal_threshold,
    calculate_psi
)
from src.ml.fraud_pipeline import engineer_fraud_features, FraudDetectionPipeline
from src.ml.risk_pipeline import clean_and_engineer_risk, CreditRiskPipeline
from src.api.app_api import app, load_pipelines

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def init_api_models():
    load_pipelines()


def test_classification_metrics():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.9])
    
    metrics = evaluate_classification_metrics(y_true, y_prob, threshold=0.5)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.2, 0.35, 0.4, 0.8, 0.9])
    
    thresh, best_f1, curve = find_optimal_threshold(y_true, y_prob, metric="f1")
    assert 0.05 <= thresh <= 0.95
    assert best_f1 > 0.5
    assert len(curve) > 0


def test_calculate_psi_stable():
    np.random.seed(42)
    expected = np.random.normal(50, 10, 1000)
    actual = np.random.normal(50, 10, 1000)
    
    psi, buckets = calculate_psi(expected, actual, num_buckets=10)
    assert psi < 0.10
    assert len(buckets) == 10


def test_calculate_psi_drift():
    np.random.seed(42)
    expected = np.random.normal(50, 10, 1000)
    actual = np.random.normal(85, 15, 1000)
    
    psi, buckets = calculate_psi(expected, actual, num_buckets=10)
    assert psi > 0.25


def test_fraud_feature_engineering():
    dummy = pd.DataFrame([{
        'merchant': 'fraud_test',
        'category': 'shopping_net',
        'amt': 120.50,
        'gender': 'M',
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 500000,
        'merch_lat': 40.7589,
        'merch_long': -73.9851,
        'trans_date_trans_time': '2026-05-15 14:30:00',
        'dob': '1990-01-01'
    }])
    
    df_eng, stats = engineer_fraud_features(dummy, is_train=True)
    assert 'distance_km' in df_eng.columns
    assert 'sin_hour' in df_eng.columns
    assert 'cos_hour' in df_eng.columns
    assert 'amt_log' in df_eng.columns
    assert 'age' in df_eng.columns
    assert df_eng['distance_km'].iloc[0] > 0


def test_risk_feature_engineering():
    dummy = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': 0.75,
        'age': 40,
        'NumberOfTime30-59DaysPastDueNotWorse': 1,
        'DebtRatio': 0.45,
        'MonthlyIncome': 5000.0,
        'NumberOfOpenCreditLinesAndLoans': 6,
        'NumberOfTimes90DaysLate': 0,
        'NumberRealEstateLoansOrLines': 1,
        'NumberOfTime60-89DaysPastDueNotWorse': 0,
        'NumberOfDependents': 2
    }])
    
    df_eng, quantiles = clean_and_engineer_risk(dummy, is_train=True)
    assert 'weighted_delinquency_score' in df_eng.columns
    assert 'disposable_income_estimate' in df_eng.columns
    assert 'income_per_dependent' in df_eng.columns
    assert df_eng['weighted_delinquency_score'].iloc[0] == 1.0


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


def test_api_predict_fraud():
    payload = {
        "merchant": "fraud_Kirlin and Sons",
        "category": "shopping_net",
        "amt": 249.99,
        "gender": "M",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "merch_lat": 33.98,
        "merch_long": -81.20,
        "trans_date_trans_time": "2026-07-24 14:35:00",
        "dob": "1985-03-19"
    }
    res = client.post("/predict/fraud", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert data["inference_latency_ms"] >= 0


def test_api_predict_risk():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.9,
        "MonthlyIncome": 3000.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    res = client.post("/predict/risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "delinquency_probability" in data
    assert "high_risk" in data
    assert "risk_tier" in data
    assert "recommended_action" in data


def test_api_explain_fraud():
    payload = {
        "merchant": "fraud_Kirlin and Sons",
        "category": "shopping_net",
        "amt": 249.99,
        "gender": "M",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "merch_lat": 33.98,
        "merch_long": -81.20,
        "trans_date_trans_time": "2026-07-24 14:35:00",
        "dob": "1985-03-19"
    }
    res = client.post("/explain/fraud", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "top_reasons" in data
    assert len(data["top_reasons"]) > 0


def test_api_explain_risk():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.9,
        "MonthlyIncome": 3000.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    res = client.post("/explain/risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "top_risk_factors" in data
    assert len(data["top_risk_factors"]) > 0
