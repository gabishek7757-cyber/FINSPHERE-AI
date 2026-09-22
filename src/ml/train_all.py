"""
FinSphere AI — Unified Training & Benchmarking Orchestrator
Loads datasets, trains Fraud and Credit Risk pipelines, evaluates baseline vs.
XGBoost with advanced feature engineering, and exports serialized production bundles.
"""
import os
import sys
import json
import shutil
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.fraud_pipeline import FraudDetectionPipeline
from src.ml.risk_pipeline import CreditRiskPipeline


def resolve_data_paths():
    fraud_train_candidates = [
        "API-layer/fraudTrain.csv",
        "datasets/fraudTrain.csv/fraudTrain.csv",
        "datasets/fraudTrain.csv",
        "fraudTrain.csv"
    ]
    fraud_test_candidates = [
        "API-layer/fraudTest.csv",
        "datasets/fraudTest.csv/fraudTest.csv",
        "datasets/fraudTest.csv",
        "fraudTest.csv"
    ]
    risk_candidates = [
        "API-layer/cs-training.csv",
        "datasets/cs-training.csv/cs-training.csv",
        "datasets/cs-training.csv",
        "cs-training.csv"
    ]

    def find_first(candidates):
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                return c
        raise FileNotFoundError(f"None of candidate paths exist: {candidates}")

    return find_first(fraud_train_candidates), find_first(fraud_test_candidates), find_first(risk_candidates)


def main():
    print("=" * 70)
    print("   FinSphere AI — Enterprise Model Training & Benchmark Pipeline   ")
    print("=" * 70)

    fraud_train_path, fraud_test_path, risk_path = resolve_data_paths()
    print(f"Data Sources:\n- Fraud Train: {fraud_train_path}\n- Fraud Test:  {fraud_test_path}\n- Credit Risk: {risk_path}\n")

    # 1. Train Credit Risk Model
    print("-" * 70)
    print(">> [1/2] Training Advanced Credit Risk Intelligence Pipeline...")
    risk_df = pd.read_csv(risk_path)
    risk_pipe = CreditRiskPipeline(output_dir="models")
    risk_metrics = risk_pipe.fit_and_evaluate(risk_df)

    # 2. Train Fraud Detection Model
    print("-" * 70)
    print(">> [2/2] Training Advanced Fraud Detection Intelligence Pipeline...")
    fraud_train_df = pd.read_csv(fraud_train_path)
    fraud_test_df = pd.read_csv(fraud_test_path)
    fraud_pipe = FraudDetectionPipeline(output_dir="models")
    # Subsample training to 350,000 for high speed and balanced representation if needed
    fraud_metrics = fraud_pipe.fit_and_evaluate(
        fraud_train_df, fraud_test_df, sample_for_training=350000
    )

    # Sync to API-layer/models for backward compatibility
    os.makedirs("API-layer/models", exist_ok=True)
    if os.path.exists("models/fraud_model.pkl"):
        shutil.copy("models/fraud_model.pkl", "API-layer/models/fraud_model.pkl")
    if os.path.exists("models/risk_model.pkl"):
        shutil.copy("models/risk_model.pkl", "API-layer/models/risk_model.pkl")

    # Save summary metrics report
    summary = {
        "fraud_detection": fraud_metrics,
        "credit_risk": risk_metrics
    }
    with open("models/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("   MODEL BENCHMARK RESULTS & QUANTITATIVE IMPACT   ")
    print("=" * 70)
    print("\n[FRAUD DETECTION MODULE]")
    print(f"  • Random Forest Baseline: F1 = {fraud_metrics['rf_baseline']['f1']:.4f} | PR-AUC = {fraud_metrics['rf_baseline']['pr_auc']:.4f} | ROC-AUC = {fraud_metrics['rf_baseline']['roc_auc']:.4f}")
    print(f"  • Optimized XGBoost:      F1 = {fraud_metrics['xgboost_optimal']['f1']:.4f} | PR-AUC = {fraud_metrics['xgboost_optimal']['pr_auc']:.4f} | ROC-AUC = {fraud_metrics['xgboost_optimal']['roc_auc']:.4f}")
    print(f"  • Optimal Decision Threshold: {fraud_metrics['best_threshold']:.3f} (Cost Optimal Threshold: {fraud_metrics['best_cost_threshold']:.3f})")

    print("\n[CREDIT RISK ASSESSMENT MODULE]")
    print(f"  • Random Forest Baseline: F1 = {risk_metrics['rf_baseline']['f1']:.4f} | PR-AUC = {risk_metrics['rf_baseline']['pr_auc']:.4f} | ROC-AUC = {risk_metrics['rf_baseline']['roc_auc']:.4f}")
    print(f"  • Optimized XGBoost:      F1 = {risk_metrics['xgboost_optimal']['f1']:.4f} | PR-AUC = {risk_metrics['xgboost_optimal']['pr_auc']:.4f} | ROC-AUC = {risk_metrics['xgboost_optimal']['roc_auc']:.4f}")
    print(f"  • Optimal Decision Threshold: {risk_metrics['best_threshold']:.3f} (Cost Optimal Threshold: {risk_metrics['best_cost_threshold']:.3f})")
    print("=" * 70)
    print("Training successfully completed. All bundles and explainers saved.")


if __name__ == "__main__":
    main()
