"""
FinSphere AI — Real-Time Prediction API

Exposes two endpoints:
  POST /predict/fraud  — score a single transaction for fraud
  POST /predict/risk   — score a single borrower for credit delinquency risk
  GET  /health          — basic liveness check

Run locally:
    uvicorn main:app --reload --port 8000

Then test with the built-in docs UI at http://localhost:8000/docs
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(
    title="FinSphere AI — Prediction API",
    description="Real-time fraud detection and credit risk scoring for FinSphere AI",
    version="1.0.0",
)

# Allow the dashboard to be opened from anywhere (e.g. a different tunnel URL, or
# opened as a local file) without the browser blocking the request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def dashboard():
    """Serves the custom risk console UI at the API's root URL."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "FinSphere AI API is running. Visit /docs for the interactive API explorer."}


MODELS_DIR = "models"
fraud_bundle = None
risk_bundle = None


@app.on_event("startup")
def load_models():
    global fraud_bundle, risk_bundle
    fraud_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
    risk_path = os.path.join(MODELS_DIR, "risk_model.pkl")

    if os.path.exists(fraud_path):
        fraud_bundle = joblib.load(fraud_path)
        print("Fraud model loaded.")
    else:
        print(f"WARNING: {fraud_path} not found. /predict/fraud will return 503 until it's trained "
              f"(run train_export_fraud.py first).")

    if os.path.exists(risk_path):
        risk_bundle = joblib.load(risk_path)
        print("Risk model loaded.")
    else:
        print(f"WARNING: {risk_path} not found. /predict/risk will return 503 until it's trained "
              f"(run train_export_risk.py first).")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    merchant: str = Field(..., example="fraud_Kirlin and Sons")
    category: str = Field(..., example="shopping_net")
    amt: float = Field(..., example=249.99)
    gender: str = Field(..., example="M")
    lat: float = Field(..., example=33.9659)
    long: float = Field(..., example=-80.9355)
    city_pop: int = Field(..., example=333497)
    merch_lat: float = Field(..., example=33.98)
    merch_long: float = Field(..., example=-81.20)
    trans_date_trans_time: str = Field(..., example="2026-07-24 14:35:00")
    dob: str = Field(..., example="1985-03-19")


class FraudPrediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float


class BorrowerInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., example=0.35)
    age: int = Field(..., example=45)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime30-59DaysPastDueNotWorse", example=0)
    DebtRatio: float = Field(..., example=0.42)
    MonthlyIncome: float = Field(..., example=5500)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., example=6)
    NumberOfTimes90DaysLate: int = Field(..., example=0)
    NumberRealEstateLoansOrLines: int = Field(..., example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime60-89DaysPastDueNotWorse", example=0)
    NumberOfDependents: int = Field(..., example=2)

    class Config:
        populate_by_name = True


class RiskPrediction(BaseModel):
    high_risk: bool
    delinquency_probability: float
    threshold_used: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "fraud_model_loaded": fraud_bundle is not None,
        "risk_model_loaded": risk_bundle is not None,
    }


@app.post("/predict/fraud", response_model=FraudPrediction)
def predict_fraud(transaction: TransactionInput):
    if fraud_bundle is None:
        raise HTTPException(status_code=503, detail="Fraud model not loaded. Run train_export_fraud.py first.")

    model = fraud_bundle["model"]
    encoders = fraud_bundle["encoders"]
    feature_cols = fraud_bundle["feature_cols"]
    threshold = fraud_bundle["threshold"]

    row = transaction.dict()
    ts = pd.to_datetime(row["trans_date_trans_time"])
    dob = pd.to_datetime(row["dob"])
    row["hour"] = ts.hour
    row["day_of_week"] = ts.dayofweek
    row["age"] = (ts - dob).days // 365

    R = 6371
    lat1, lon1 = np.radians(row["lat"]), np.radians(row["long"])
    lat2, lon2 = np.radians(row["merch_lat"]), np.radians(row["merch_long"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    row["distance_km"] = R * 2 * np.arcsin(np.sqrt(a))
    row["amt_log"] = np.log1p(row["amt"])

    for col in ["merchant", "category", "gender"]:
        le = encoders[col]
        known = set(le.classes_)
        val = row[col] if row[col] in known else le.classes_[0]
        row[col] = le.transform([val])[0]

    X = pd.DataFrame([{c: row[c] for c in feature_cols}])
    proba = float(model.predict_proba(X)[0, 1])

    return FraudPrediction(
        is_fraud=proba >= threshold,
        fraud_probability=round(proba, 6),
        threshold_used=threshold,
    )


@app.post("/predict/risk", response_model=RiskPrediction)
def predict_risk(borrower: BorrowerInput):
    if risk_bundle is None:
        raise HTTPException(status_code=503, detail="Risk model not loaded. Run train_export_risk.py first.")

    model = risk_bundle["model"]
    feature_cols = risk_bundle["feature_cols"]
    late_cols = risk_bundle["late_cols"]
    threshold = risk_bundle["threshold"]

    row = borrower.dict(by_alias=True)
    row["RevolvingUtilizationOfUnsecuredLines"] = min(row["RevolvingUtilizationOfUnsecuredLines"], 2.0)
    row["DebtRatio"] = min(row["DebtRatio"], 5.0)
    for col in late_cols:
        row[col] = min(row[col], 18)

    row["total_past_due"] = sum(row[c] for c in late_cols)
    row["income_per_dependent"] = row["MonthlyIncome"] / (row["NumberOfDependents"] + 1)

    X = pd.DataFrame([{c: row[c] for c in feature_cols}])
    proba = float(model.predict_proba(X)[0, 1])

    return RiskPrediction(
        high_risk=proba >= threshold,
        delinquency_probability=round(proba, 6),
        threshold_used=threshold,
    )
