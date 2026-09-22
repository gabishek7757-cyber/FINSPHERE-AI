"""
FinSphere AI — Advanced Fraud Detection Pipeline
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.ml.metrics_utils import evaluate_classification_metrics, find_optimal_threshold


def engineer_fraud_features(df, category_stats=None, is_train=False):
    df = df.copy()
    
    ts = pd.to_datetime(df['trans_date_trans_time'])
    dob = pd.to_datetime(df['dob'])
    
    df['hour'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek
    df['age'] = ((ts - dob).dt.days // 365).clip(18, 100)
    
    # Cyclical hour encoding
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Haversine distance
    R = 6371.0
    lat1, lon1 = np.radians(df['lat'].values), np.radians(df['long'].values)
    lat2, lon2 = np.radians(df['merch_lat'].values), np.radians(df['merch_long'].values)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df['distance_km'] = R * 2.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1.0))
    
    df['amt_log'] = np.log1p(df['amt'])
    
    if is_train:
        cat_means = df.groupby('category')['amt'].agg(['mean', 'std']).reset_index()
        cat_means.columns = ['category', 'cat_amt_mean', 'cat_amt_std']
        cat_means['cat_amt_std'] = cat_means['cat_amt_std'].fillna(1.0)
        category_stats = cat_means.set_index('category').to_dict('index')
    
    if category_stats:
        cat_mean_map = {k: v['cat_amt_mean'] for k, v in category_stats.items()}
        cat_std_map = {k: v['cat_amt_std'] for k, v in category_stats.items()}
        default_mean = float(np.mean(list(cat_mean_map.values()))) if cat_mean_map else 50.0
        default_std = float(np.mean(list(cat_std_map.values()))) if cat_std_map else 30.0
        
        cat_mean_col = df['category'].map(cat_mean_map).fillna(default_mean)
        cat_std_col = df['category'].map(cat_std_map).fillna(default_std).replace(0, 1.0)
        
        df['amt_to_cat_mean_ratio'] = df['amt'] / (cat_mean_col + 1e-4)
        df['amt_cat_zscore'] = (df['amt'] - cat_mean_col) / cat_std_col
    else:
        df['amt_to_cat_mean_ratio'] = 1.0
        df['amt_cat_zscore'] = 0.0
        
    return df, category_stats


class FraudDetectionPipeline:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        self.encoders = {}
        self.category_stats = {}
        self.feature_cols = [
            'merchant', 'category', 'amt', 'amt_log', 'gender',
            'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'hour', 'day_of_week', 'age', 'distance_km',
            'sin_hour', 'cos_hour', 'is_night', 'is_weekend',
            'amt_to_cat_mean_ratio', 'amt_cat_zscore'
        ]
        self.model = None
        self.explainer = None
        self.best_threshold = 0.45
        self.metrics_report = {}

    def fit_and_evaluate(self, train_df, test_df, sample_for_training=None):
        print("[Fraud Pipeline] Engineering features for train and test...")
        if sample_for_training and len(train_df) > sample_for_training:
            print(f"[Fraud Pipeline] Subsampling training set to {sample_for_training:,} for performance...")
            fraud_rows = train_df[train_df['is_fraud'] == 1]
            non_fraud_rows = train_df[train_df['is_fraud'] == 0].sample(
                n=sample_for_training - len(fraud_rows), random_state=42
            )
            train_subset = pd.concat([fraud_rows, non_fraud_rows]).sample(frac=1.0, random_state=42)
        else:
            train_subset = train_df

        train_eng, self.category_stats = engineer_fraud_features(train_subset, is_train=True)
        test_eng, _ = engineer_fraud_features(test_df, category_stats=self.category_stats, is_train=False)

        cat_cols = ['merchant', 'category', 'gender']
        for c in cat_cols:
            le = LabelEncoder()
            train_eng[c] = le.fit_transform(train_eng[c].astype(str))
            known = set(le.classes_)
            test_eng[c] = test_eng[c].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            test_eng[c] = le.transform(test_eng[c])
            self.encoders[c] = le

        X_train = train_eng[self.feature_cols].fillna(0)
        y_train = train_eng['is_fraud'].values
        X_test = test_eng[self.feature_cols].fillna(0)
        y_test = test_eng['is_fraud'].values

        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_weight = float(neg_count / max(pos_count, 1))

        print(f"[Fraud Pipeline] Training Advanced XGBoost Classifier...")
        self.model = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=min(scale_weight, 25.0),
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        self.best_threshold, best_f1, f1_curve = find_optimal_threshold(y_test, y_prob, metric='f1')
        best_cost_threshold, min_cost, cost_curve = find_optimal_threshold(
            y_test, y_prob, metric='cost', cost_fp=15.0, cost_fn=150.0
        )
        
        default_metrics = evaluate_classification_metrics(y_test, y_prob, threshold=0.5)
        optimal_metrics = evaluate_classification_metrics(y_test, y_prob, threshold=self.best_threshold)

        print(f"[Fraud Pipeline] XGBoost F1: {optimal_metrics['f1']:.4f} (thresh={self.best_threshold:.3f}) | PR-AUC: {optimal_metrics['pr_auc']:.4f} | ROC-AUC: {optimal_metrics['roc_auc']:.4f}")

        print("[Fraud Pipeline] Training baseline Random Forest for benchmark comparison...")
        rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=14, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
        rf_baseline.fit(X_train, y_train)
        rf_prob = rf_baseline.predict_proba(X_test)[:, 1]
        rf_opt_thresh, _, _ = find_optimal_threshold(y_test, rf_prob, metric='f1')
        rf_metrics = evaluate_classification_metrics(y_test, rf_prob, threshold=rf_opt_thresh)

        print("[Fraud Pipeline] Building TreeSHAP Explainer...")
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
        df = pd.DataFrame([raw_dict])
        df_eng, _ = engineer_fraud_features(df, category_stats=self.category_stats, is_train=False)
        
        for c in ['merchant', 'category', 'gender']:
            le = self.encoders[c]
            known = set(le.classes_)
            val = df_eng[c].iloc[0] if df_eng[c].iloc[0] in known else le.classes_[0]
            df_eng[c] = le.transform([val])[0]
            
        X = df_eng[self.feature_cols].fillna(0)
        return X

    def predict(self, raw_dict, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        X = self.transform_single(raw_dict)
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            'is_fraud': bool(prob >= threshold),
            'fraud_probability': round(prob, 6),
            'threshold_used': float(threshold),
            'risk_level': 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= threshold else 'LOW')
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
                'impact': 'Increases Fraud Risk' if s_val > 0 else 'Reduces Fraud Risk'
            })
            
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return {
            'base_value': round(base_val, 5),
            'prediction': self.predict(raw_dict),
            'top_reasons': contributions[:8],
            'all_contributions': contributions
        }

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'fraud_model.pkl')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'model': self.model,
            'encoders': self.encoders,
            'category_stats': self.category_stats,
            'feature_cols': self.feature_cols,
            'threshold': self.best_threshold,
            'explainer': self.explainer,
            'metrics_report': self.metrics_report
        }
        joblib.dump(bundle, filepath)
        print(f"[Fraud Pipeline] Model bundle saved to {filepath}")

    @classmethod
    def load(cls, filepath='models/fraud_model.pkl'):
        bundle = joblib.load(filepath)
        instance = cls()
        instance.model = bundle['model']
        instance.encoders = bundle['encoders']
        instance.category_stats = bundle.get('category_stats', {})
        instance.feature_cols = bundle['feature_cols']
        instance.best_threshold = bundle.get('threshold', 0.45)
        instance.explainer = bundle.get('explainer')
        instance.metrics_report = bundle.get('metrics_report', {})
        return instance
