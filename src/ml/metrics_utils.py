"""
FinSphere AI — Advanced Metrics & Operational Utilities
Includes: Comprehensive Classification Evaluation, Optimal Financial Threshold Search,
Expected Dollar Loss Matrix, and Population Stability Index (PSI) Drift Monitoring.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)


def evaluate_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute a complete battery of classification metrics for imbalanced regimes.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    }


def find_optimal_threshold(y_true, y_prob, metric="f1", cost_fp=15.0, cost_fn=150.0):
    """
    Scans probability thresholds from 0.05 to 0.95 in 0.025 increments.
    Can optimize for F1-score, Recall, or Business Expected Cost.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.arange(0.05, 0.96, 0.025)
    best_thresh = 0.5
    best_score = -float("inf") if metric != "cost" else float("inf")
    curve = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        cost = (fp * cost_fp) + (fn * cost_fn)

        record = {
            "threshold": round(float(t), 3),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "cost": round(float(cost), 2),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }
        curve.append(record)

        if metric == "f1" and f1 > best_score:
            best_score = f1
            best_thresh = t
        elif metric == "cost" and cost < best_score:
            best_score = cost
            best_thresh = t
        elif metric == "recall" and rec > best_score:
            best_score = rec
            best_thresh = t

    return round(float(best_thresh), 3), round(float(best_score), 4), curve


def calculate_psi(expected_series, actual_series, num_buckets=10):
    """
    Calculate the Population Stability Index (PSI) between baseline and production data.
    PSI < 0.1: No significant shift (Stable)
    0.1 <= PSI < 0.25: Moderate shift (Monitor)
    PSI >= 0.25: Significant shift (Action/Retrain Required)
    """
    expected = np.asarray(pd.Series(expected_series).dropna())
    actual = np.asarray(pd.Series(actual_series).dropna())
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, []

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    psi_total = 0.0
    bucket_records = []

    for i in range(num_buckets):
        low, high = bucket_bounds[i], bucket_bounds[i+1]
        exp_count = np.sum((expected > low) & (expected <= high))
        act_count = np.sum((actual > low) & (actual <= high))

        exp_pct = max(exp_count / len(expected), 1e-4)
        act_pct = max(act_count / len(actual), 1e-4)

        bucket_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        psi_total += bucket_psi

        bucket_records.append({
            "bucket": f"B{i+1}",
            "range": f"[{low:.2f}, {high:.2f}]",
            "expected_pct": round(exp_pct * 100, 2),
            "actual_pct": round(act_pct * 100, 2),
            "psi_contribution": round(bucket_psi, 4)
        })

    return round(float(psi_total), 4), bucket_records
