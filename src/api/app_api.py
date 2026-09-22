"""
FinSphere AI — Enterprise Real-Time Prediction & Explainability API
"""
import os
import sys
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.fraud_pipeline import FraudDetectionPipeline
from src.ml.risk_pipeline import CreditRiskPipeline

app = FastAPI(
    title="FinSphere AI — Enterprise Intelligence API",
    description="Production-grade real-time fraud detection, credit risk scoring, and Explainable AI (SHAP) services.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("API-layer/static"):
    app.mount("/static", StaticFiles(directory="API-layer/static"), name="static")
elif os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

fraud_pipe: Optional[FraudDetectionPipeline] = None
risk_pipe: Optional[CreditRiskPipeline] = None


@app.on_event("startup")
def load_pipelines():
    global fraud_pipe, risk_pipe
    fraud_paths = ["models/fraud_model.pkl", "API-layer/models/fraud_model.pkl"]
    risk_paths = ["models/risk_model.pkl", "API-layer/models/risk_model.pkl"]

    for p in fraud_paths:
        if os.path.exists(p):
            try:
                fraud_pipe = FraudDetectionPipeline.load(p)
                print(f"[API] Loaded Fraud model from {p}")
                break
            except Exception as e:
                print(f"[API] Error loading fraud model from {p}: {e}")

    for p in risk_paths:
        if os.path.exists(p):
            try:
                risk_pipe = CreditRiskPipeline.load(p)
                print(f"[API] Loaded Risk model from {p}")
                break
            except Exception as e:
                print(f"[API] Error loading risk model from {p}: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    merchant: str = Field(..., json_schema_extra={"example": "fraud_Kirlin and Sons"})
    category: str = Field(..., json_schema_extra={"example": "shopping_net"})
    amt: float = Field(..., json_schema_extra={"example": 249.99})
    gender: str = Field(..., json_schema_extra={"example": "M"})
    lat: float = Field(..., json_schema_extra={"example": 33.9659})
    long: float = Field(..., json_schema_extra={"example": -80.9355})
    city_pop: int = Field(..., json_schema_extra={"example": 333497})
    merch_lat: float = Field(..., json_schema_extra={"example": 33.98})
    merch_long: float = Field(..., json_schema_extra={"example": -81.20})
    trans_date_trans_time: str = Field(..., json_schema_extra={"example": "2026-07-24 14:35:00"})
    dob: str = Field(..., json_schema_extra={"example": "1985-03-19"})
    custom_threshold: Optional[float] = Field(None, json_schema_extra={"example": 0.45})


class FraudPredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    risk_level: str
    inference_latency_ms: float


class BorrowerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(..., json_schema_extra={"example": 0.85})
    age: int = Field(..., json_schema_extra={"example": 45})
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime30-59DaysPastDueNotWorse", json_schema_extra={"example": 2})
    DebtRatio: float = Field(..., json_schema_extra={"example": 0.90})
    MonthlyIncome: float = Field(..., json_schema_extra={"example": 3200.0})
    NumberOfOpenCreditLinesAndLoans: int = Field(..., json_schema_extra={"example": 6})
    NumberOfTimes90DaysLate: int = Field(..., json_schema_extra={"example": 1})
    NumberRealEstateLoansOrLines: int = Field(..., json_schema_extra={"example": 1})
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime60-89DaysPastDueNotWorse", json_schema_extra={"example": 0})
    NumberOfDependents: int = Field(..., json_schema_extra={"example": 2})
    custom_threshold: Optional[float] = Field(None, json_schema_extra={"example": 0.35})


class RiskPredictionResponse(BaseModel):
    high_risk: bool
    delinquency_probability: float
    threshold_used: float
    risk_tier: str
    recommended_action: str
    inference_latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    if os.path.exists("API-layer/static/index.html"):
        return FileResponse("API-layer/static/index.html")
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "FinSphere AI v2.0 Enterprise Intelligence API is running. Visit /docs for API documentation."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "fraud_model_ready": fraud_pipe is not None,
        "risk_model_ready": risk_pipe is not None,
    }


@app.get("/metrics")
def metrics():
    return {
        "fraud_detection": fraud_pipe.metrics_report if fraud_pipe else {},
        "credit_risk": risk_pipe.metrics_report if risk_pipe else {}
    }


@app.post("/predict/fraud", response_model=FraudPredictionResponse)
def predict_fraud(transaction: TransactionInput):
    if fraud_pipe is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded. Run train_all.py first.")
    
    t0 = time.perf_counter()
    raw = transaction.model_dump(exclude={"custom_threshold"})
    result = fraud_pipe.predict(raw, threshold=transaction.custom_threshold)
    latency = (time.perf_counter() - t0) * 1000.0
    
    return FraudPredictionResponse(
        is_fraud=result["is_fraud"],
        fraud_probability=result["fraud_probability"],
        threshold_used=result["threshold_used"],
        risk_level=result["risk_level"],
        inference_latency_ms=round(latency, 2)
    )


@app.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(borrower: BorrowerInput):
    if risk_pipe is None:
        raise HTTPException(status_code=503, detail="Credit risk model not loaded. Run train_all.py first.")
    
    t0 = time.perf_counter()
    raw = borrower.model_dump(by_alias=True, exclude={"custom_threshold"})
    result = risk_pipe.predict(raw, threshold=borrower.custom_threshold)
    latency = (time.perf_counter() - t0) * 1000.0
    
    return RiskPredictionResponse(
        high_risk=result["high_risk"],
        delinquency_probability=result["delinquency_probability"],
        threshold_used=result["threshold_used"],
        risk_tier=result["risk_tier"],
        recommended_action=result["recommended_action"],
        inference_latency_ms=round(latency, 2)
    )


@app.post("/explain/fraud")
def explain_fraud(transaction: TransactionInput):
    if fraud_pipe is None or fraud_pipe.explainer is None:
        raise HTTPException(status_code=503, detail="Fraud Explainer not available.")
    
    raw = transaction.model_dump(exclude={"custom_threshold"})
    explanation = fraud_pipe.explain(raw)
    return explanation


@app.post("/explain/risk")
def explain_risk(borrower: BorrowerInput):
    if risk_pipe is None or risk_pipe.explainer is None:
        raise HTTPException(status_code=503, detail="Risk Explainer not available.")
    
    raw = borrower.model_dump(by_alias=True, exclude={"custom_threshold"})
    explanation = risk_pipe.explain(raw)
    return explanation


@app.post("/predict/batch/fraud")
def batch_fraud(transactions: List[TransactionInput]):
    if fraud_pipe is None:
        raise HTTPException(status_code=503, detail="Fraud model not loaded.")
    
    results = []
    t0 = time.perf_counter()
    for item in transactions:
        raw = item.model_dump(exclude={"custom_threshold"})
        res = fraud_pipe.predict(raw, threshold=item.custom_threshold)
        results.append(res)
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    
    fraud_count = sum(1 for r in results if r["is_fraud"])
    return {
        "total_records": len(results),
        "flagged_fraud_count": fraud_count,
        "fraud_percentage": round(fraud_count / max(len(results), 1) * 100, 2),
        "batch_latency_ms": round(total_time_ms, 2),
        "avg_latency_per_item_ms": round(total_time_ms / max(len(results), 1), 2),
        "predictions": results
    }


@app.post("/predict/batch/risk")
def batch_risk(borrowers: List[BorrowerInput]):
    if risk_pipe is None:
        raise HTTPException(status_code=503, detail="Risk model not loaded.")
    
    results = []
    t0 = time.perf_counter()
    for item in borrowers:
        raw = item.model_dump(by_alias=True, exclude={"custom_threshold"})
        res = risk_pipe.predict(raw, threshold=item.custom_threshold)
        results.append(res)
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    
    high_risk_count = sum(1 for r in results if r["high_risk"])
    return {
        "total_records": len(results),
        "high_risk_count": high_risk_count,
        "high_risk_percentage": round(high_risk_count / max(len(results), 1) * 100, 2),
        "batch_latency_ms": round(total_time_ms, 2),
        "avg_latency_per_item_ms": round(total_time_ms / max(len(results), 1), 2),
        "predictions": results
    }
