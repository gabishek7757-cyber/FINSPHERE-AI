"""
FinSphere AI — Advanced Credit Risk & Delinquency Prediction Pipeline
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.ml.metrics_utils import evaluate_classification_metrics, find_optimal_threshold

LATE_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]


def clean_and_engineer_risk(df, quantiles=None, is_train=False):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df[df['age'] > 0] if 'SeriousDlqin2yrs' in df.columns else df
    df['age'] = df['age'].clip(lower=18, upper=100)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0).clip(0, 20)
    
    if is_train:
        income_median = float(df['MonthlyIncome'].median())
        util_q99 = float(df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99))
        debt_q99 = float(df['DebtRatio'].quantile(0.99))
        quantiles = {'income_median': income_median, 'util_q99': util_q99, 'debt_q99': debt_q99}
    
    income_med = quantiles.get('income_median', 5400.0) if quantiles else 5400.0
    util_cap = quantiles.get('util_q99', 2.0) if quantiles else 2.0
    debt_cap = quantiles.get('debt_q99', 5.0) if quantiles else 5.0

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_med).clip(lower=0)
    for col in LATE_COLS:
        df[col] = df[col].clip(lower=0, upper=18)
        
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(lower=0, upper=util_cap)
    df['DebtRatio'] = df['DebtRatio'].clip(lower=0, upper=debt_cap)
    
    # Financial Stress Features
    df['total_past_due'] = df[LATE_COLS].sum(axis=1)
    df['weighted_delinquency_score'] = (
        1.0 * df['NumberOfTime30-59DaysPastDueNotWorse'] +
        2.0 * df['NumberOfTime60-89DaysPastDueNotWorse'] +
        3.5 * df['NumberOfTimes90DaysLate']
    )
    df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1.0)
    df['disposable_income_estimate'] = df['MonthlyIncome'] * (1.0 - np.clip(df['DebtRatio'], 0, 0.95))
    df['utilization_stress_index'] = (
        (df['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int) +
        (df['RevolvingUtilizationOfUnsecuredLines'] > 1.0).astype(int)
    )
    df['real_estate_to_open_lines_ratio'] = df['NumberRealEstateLoansOrLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1.0)
    
    return df, quantiles


class CreditRiskPipeline:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        self.quantiles = {}
        self.feature_cols = [
            'RevolvingUtilizationOfUnsecuredLines', 'age',
            'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
            'total_past_due', 'weighted_delinquency_score',
            'income_per_dependent', 'disposable_income_estimate',
            'utilization_stress_index', 'real_estate_to_open_lines_ratio'
        ]
        self.model = None
        self.explainer = None
        self.best_threshold = 0.35
        self.metrics_report = {}

    def fit_and_evaluate(self, raw_df):
        print("[Credit Risk Pipeline] Cleaning data and engineering financial features...")
        cleaned_df, self.quantiles = clean_and_engineer_risk(raw_df, is_train=True)
        
        y = cleaned_df['SeriousDlqin2yrs'].values
        X = cleaned_df[self.feature_cols].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_weight = float(neg_count / max(pos_count, 1))

        print(f"[Credit Risk Pipeline] Training Advanced XGBoost Classifier (scale_pos_weight={scale_weight:.2f})...")
        self.model = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=min(scale_weight, 10.0),
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.best_threshold, best_f1, f1_curve = find_optimal_threshold(y_test, y_prob, metric='f1')
        best_cost_threshold, min_cost, cost_curve = find_optimal_threshold(
            y_test, y_prob, metric='cost', cost_fp=50.0, cost_fn=500.0
        )

        default_metrics = evaluate_classification_metrics(y_test, y_prob, threshold=0.5)
        optimal_metrics = evaluate_classification_metrics(y_test, y_prob, threshold=self.best_threshold)

        print(f"[Credit Risk Pipeline] XGBoost F1: {optimal_metrics['f1']:.4f} (thresh={self.best_threshold:.3f}) | PR-AUC: {optimal_metrics['pr_auc']:.4f} | ROC-AUC: {optimal_metrics['roc_auc']:.4f}")

        print("[Credit Risk Pipeline] Training baseline Random Forest for benchmark comparison...")
        rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
        rf_baseline.fit(X_train, y_train)
        rf_prob = rf_baseline.predict_proba(X_test)[:, 1]
        rf_opt_thresh, _, _ = find_optimal_threshold(y_test, rf_prob, metric='f1')
        rf_metrics = evaluate_classification_metrics(y_test, rf_prob, threshold=rf_opt_thresh)

        print("[Credit Risk Pipeline] Building TreeSHAP Explainer...")
        self.explainer = shap.TreeExplainer(self.model, feature_perturbation='tree_path_dependent')

        self.metrics_report = {
            'xgboost_optimal': optimal_metrics,
            'xgboost_default': default_metrics,
            'rf_baseline': rf_metrics,
            'best_threshold': self.best_threshold,
            'best_cost_threshold': best_cost_threshold,
            'f1_curve': f1_curve[:20],
            'cost_curve': cost_curve[:20],
            'feature_importances': dict(zip(self.feature_cols, [float(x) for x in self.model.feature_importances_]))
        }

        self.save()
        return self.metrics_report

    def transform_single(self, raw_dict):
        normalized = {}
        for k, v in raw_dict.items():
            clean_k = k.replace('_', '-')
            if clean_k in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse']:
                normalized[clean_k] = v
            else:
                normalized[k] = v

        df = pd.DataFrame([normalized])
        df_eng, _ = clean_and_engineer_risk(df, quantiles=self.quantiles, is_train=False)
        X = df_eng[self.feature_cols].fillna(0)
        return X

    def predict(self, raw_dict, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        X = self.transform_single(raw_dict)
        prob = float(self.model.predict_proba(X)[0, 1])
        
        if prob >= 0.65:
            tier = "HIGH_RISK_REJECT"
            action = "Decline application or require collateral"
        elif prob >= threshold:
            tier = "MODERATE_RISK_REVIEW"
            action = "Manual underwriter review required"
        else:
            tier = "LOW_RISK_APPROVE"
            action = "Standard approval eligible"

        return {
            'high_risk': bool(prob >= threshold),
            'delinquency_probability': round(prob, 6),
            'threshold_used': float(threshold),
            'risk_tier': tier,
            'recommended_action': action
        }

    def explain(self, raw_dict):
        X = self.transform_single(raw_dict)
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model, feature_perturbation='tree_path_dependent')
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
            
        values = shap_vals[0]
        base_val = float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[-1])

        contributions = []
        for feat, val, s_val in zip(self.feature_cols, X.iloc[0], values):
            contributions.append({
                'feature': feat,
                'feature_value': float(val) if isinstance(val, (int, float, np.number)) else str(val),
                'shap_value': round(float(s_val), 5),
                'impact': 'Increases Delinquency Risk' if s_val > 0 else 'Reduces Delinquency Risk'
            })

        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return {
            'base_value': round(base_val, 5),
            'prediction': self.predict(raw_dict),
            'top_risk_factors': contributions[:6],
            'all_contributions': contributions
        }

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'risk_model.pkl')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'model': self.model,
            'quantiles': self.quantiles,
            'feature_cols': self.feature_cols,
            'late_cols': LATE_COLS,
            'threshold': self.best_threshold,
            'explainer': self.explainer,
            'metrics_report': self.metrics_report
        }
        joblib.dump(bundle, filepath)
        print(f"[Credit Risk Pipeline] Model bundle saved to {filepath}")

    @classmethod
    def load(cls, filepath='models/risk_model.pkl'):
        bundle = joblib.load(filepath)
        instance = cls()
        instance.model = bundle['model']
        instance.quantiles = bundle.get('quantiles', {})
        instance.feature_cols = bundle['feature_cols']
        instance.best_threshold = bundle.get('threshold', 0.35)
        instance.explainer = bundle.get('explainer')
        instance.metrics_report = bundle.get('metrics_report', {})
        return instance
