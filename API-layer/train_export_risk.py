"""
Trains the credit risk Random Forest model on cs-training.csv and exports
everything the API needs as a single joblib bundle: models/risk_model.pkl

Run this once (in Colab or locally) after cs-training.csv is present in the
same folder, or update the path below.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

DATA_PATH = "cs-training.csv"
OUTPUT_PATH = "models/risk_model.pkl"

LATE_COLS = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
             'NumberOfTime60-89DaysPastDueNotWorse']


def clean_and_engineer(df):
    df = df.copy()
    df = df[df['age'] > 0]
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

    for col in LATE_COLS:
        df[col] = df[col].clip(upper=18)

    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(
        upper=df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99))
    df['DebtRatio'] = df['DebtRatio'].clip(upper=df['DebtRatio'].quantile(0.99))

    df['total_past_due'] = df[LATE_COLS].sum(axis=1)
    df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    return df


def main():
    print("Loading data...")
    risk = pd.read_csv(DATA_PATH).drop(columns=['Unnamed: 0'])
    risk = clean_and_engineer(risk)

    y = risk['SeriousDlqin2yrs']
    X = risk.drop(columns=['SeriousDlqin2yrs'])
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
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
        "feature_cols": feature_cols,
        "late_cols": LATE_COLS,
        "threshold": 0.5,
    }
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"Saved bundle to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
