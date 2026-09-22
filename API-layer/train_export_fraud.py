"""
Trains the fraud detection Random Forest model on fraudTrain.csv / fraudTest.csv
and exports everything the API needs (model, encoders, feature list) as a single
joblib bundle: models/fraud_model.pkl

Run this once (in Colab or locally) after fraudTrain.csv / fraudTest.csv are present
in the same folder, or update the paths below.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

TRAIN_PATH = "fraudTrain.csv"
TEST_PATH = "fraudTest.csv"
OUTPUT_PATH = "models/fraud_model.pkl"


def engineer_features(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    R = 6371
    lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
    lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df['distance_km'] = R * 2 * np.arcsin(np.sqrt(a))
    df['amt_log'] = np.log1p(df['amt'])
    return df


def main():
    print("Loading data...")
    fraud_train = pd.read_csv(TRAIN_PATH)
    fraud_test = pd.read_csv(TEST_PATH)

    fraud_train = engineer_features(fraud_train)
    fraud_test = engineer_features(fraud_test)

    categorical_cols = ['merchant', 'category', 'gender']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        fraud_train[col] = le.fit_transform(fraud_train[col].astype(str))
        known = set(le.classes_)
        fraud_test[col] = fraud_test[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        fraud_test[col] = le.transform(fraud_test[col])
        encoders[col] = le

    feature_cols = [
        'merchant', 'category', 'amt', 'amt_log', 'gender',
        'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
        'hour', 'day_of_week', 'age', 'distance_km'
    ]

    X_train = fraud_train[feature_cols].fillna(0)
    y_train = fraud_train['is_fraud']
    X_test = fraud_test[feature_cols].fillna(0)
    y_test = fraud_test['is_fraud']

    mask_train, mask_test = y_train.notna(), y_test.notna()
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=2,
        class_weight='balanced_subsample', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}  PR-AUC: {average_precision_score(y_test, proba):.4f}")

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": model,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "threshold": 0.45,  # from threshold-tuning step in the notebook
    }
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"Saved bundle to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
