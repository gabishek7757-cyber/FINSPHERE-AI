# FinSphere AI — Enterprise Financial Intelligence Platform

> **End-to-End ML Engineering Portfolio Project** | Fraud Detection + Credit Risk Underwriting  
> Real-time predictions • Explainable AI (SHAP) • FastAPI production service • Interactive Streamlit dashboard • Dockerized

---

## Architecture Overview

```
finsphere-ai/
├── src/
│   ├── ml/
│   │   ├── fraud_pipeline.py      # XGBoost fraud detection + TreeSHAP
│   │   ├── risk_pipeline.py       # XGBoost credit risk + TreeSHAP
│   │   ├── train_all.py           # Unified training orchestrator
│   │   └── metrics_utils.py       # Evaluation, threshold tuning, PSI drift
│   ├── api/
│   │   └── app_api.py             # FastAPI: predict, explain, batch, metrics
│   └── app/
│       └── streamlit_app.py       # Full-stack interactive web app
├── notebooks/
│   ├── fraud_detection_improved.ipynb    # XGBoost + SHAP analysis notebook
│   └── credit_risk_model.ipynb           # Credit risk underwriting notebook
├── API-layer/
│   ├── main.py                    # Legacy FastAPI (v1, retained)
│   ├── train_export_fraud.py
│   ├── train_export_risk.py
│   └── static/index.html          # Risk console UI
├── tests/
│   └── test_pipelines.py          # Comprehensive pytest suite (11 tests)
├── models/                        # Serialized model bundles + benchmark JSON
├── Dockerfile
└── requirements.txt
```

---

## Benchmark Results

### Fraud Detection (1.29M transactions, 0.57% fraud rate)

| Model | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|
| Random Forest Baseline | 0.7759 | 0.8274 | 0.9953 |
| **Optimized XGBoost + Feature Engineering** | **0.8132** | **0.8781** | **0.9974** |

**+3.7pp F1-Score improvement** | **+5.1pp PR-AUC improvement** | Threshold tuned from 0.5 → 0.825 via financial loss matrix

### Credit Risk Assessment (150K borrowers, 6.7% delinquency rate)

| Model | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|
| Random Forest Baseline | 0.4352 | 0.3743 | 0.8619 |
| **Optimized XGBoost + Stress Index Engineering** | **0.4497** | **0.3987** | **0.8686** |

**+1.5pp F1-Score improvement** | **+2.4pp PR-AUC improvement** | Regulatory SHAP adverse action notice support

---

## Key ML Engineering Highlights

### Advanced Feature Engineering
- **Cyclical Temporal Encoding**: `sin_hour` + `cos_hour` (eliminates discontinuity at midnight vs. integer hour)
- **Haversine Geodesic Distance**: Customer-to-merchant physical distance in km
- **Category Amount Deviation (Z-score)**: Transaction amount anomaly relative to merchant category historical baseline
- **Weighted Delinquency Severity**: `1.0×(30-59d) + 2.0×(60-89d) + 3.5×(90+d)` — weights reflect credit bureau severity tiers
- **Disposable Income Estimate**: `MonthlyIncome × (1 - DebtRatio)` — real financial cushion approximation

### Model Architecture
- **XGBoost** with `scale_pos_weight` for cost-sensitive imbalance handling
- `tree_method='hist'` for memory-efficient histogram-based training on large datasets
- Stratified train/test split preserving minority class proportions
- Optimal decision thresholding via both F1-maximization and Financial Cost Matrix

### Explainable AI
- **TreeSHAP** local feature attribution for every prediction
- Real-time SHAP waterfall charts rendered in the Streamlit dashboard
- Regulatory-compliant adverse action notice generation for credit decisions

### Production API (FastAPI)
- `POST /predict/fraud` — Real-time transaction scoring with risk tiers (LOW/MEDIUM/HIGH)
- `POST /predict/risk` — Credit delinquency scoring + underwriting decision tiers
- `POST /explain/fraud` — Local TreeSHAP explanation for any transaction
- `POST /explain/risk` — Adverse action factors for any loan application
- `POST /predict/batch/fraud` — Batch CSV scoring endpoint
- `GET /metrics` — Live model performance benchmarks
- Sub-10ms inference latency per prediction

### Data Governance
- **Population Stability Index (PSI)** drift monitoring module for production feature shift detection
- Benchmark metrics serialized to `models/benchmark_summary.json` for auditing

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (First-Time Setup)
```bash
python src/ml/train_all.py
```
This trains both models, computes SHAP explainers, and saves all bundles to `models/`.

### 3. Start Production API
```bash
uvicorn src.api.app_api:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for the interactive API explorer.

### 4. Launch Streamlit Dashboard
```bash
streamlit run src/app/streamlit_app.py
```

### 5. Run via Docker
```bash
docker build -t finsphere-ai .
docker run -p 8000:8000 finsphere-ai
```

### 6. Run Tests
```bash
pytest -v tests/
```

---

## API Usage Examples

### Score a Transaction for Fraud
```bash
curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "merchant": "fraud_Kirlin and Sons",
    "category": "shopping_net",
    "amt": 899.00,
    "gender": "M",
    "lat": 33.9659,
    "long": -80.9355,
    "city_pop": 333497,
    "merch_lat": 38.50,
    "merch_long": -120.30,
    "trans_date_trans_time": "2026-07-24 02:45:00",
    "dob": "1985-03-19"
  }'
```
Response: `{"is_fraud": true, "fraud_probability": 0.9872, "threshold_used": 0.825, "risk_level": "HIGH", "inference_latency_ms": 4.2}`

### Get SHAP Explanation for a Credit Decision
```bash
curl -X POST http://localhost:8000/explain/risk \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.98,
    "age": 29,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.88,
    "MonthlyIncome": 2500,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 2,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 0
  }'
```

---

## Datasets
- **Fraud Detection**: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — 1.29M transactions
- **Credit Risk**: [Give Me Some Credit (Kaggle)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) — 150K borrowers

> Note: CSVs are excluded from the repository via `.gitignore`. Place data files in `API-layer/` before running training.

---

## Technologies Used
`Python 3.11` · `XGBoost` · `Scikit-Learn` · `SHAP` · `FastAPI` · `Streamlit` · `Pandas` · `NumPy` · `Docker` · `pytest`
