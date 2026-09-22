# FinSphere AI — Real-Time Prediction API

This turns your two trained models (fraud detection + credit risk) into a live REST API
that a dashboard or a bank application can call in real time — the piece your Slide 8
"Challenges" flagged (linking the model to a real-time system).

Tested end-to-end and confirmed working: both `/predict/fraud` and `/predict/risk`
return correct predictions.

## Files

| File | Purpose |
|---|---|
| `train_export_fraud.py` | Trains the fraud model on `fraudTrain.csv`/`fraudTest.csv`, saves `models/fraud_model.pkl` |
| `train_export_risk.py` | Trains the credit risk model on `cs-training.csv`, saves `models/risk_model.pkl` |
| `main.py` | The FastAPI app — loads both `.pkl` bundles and serves predictions |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and export both models
Put `fraudTrain.csv`, `fraudTest.csv`, and `cs-training.csv` in this same folder, then:
```bash
python train_export_fraud.py
python train_export_risk.py
```
This creates `models/fraud_model.pkl` and `models/risk_model.pkl`. You only need to do
this once (retrain if you improve the models later).

### 3. Run the API
```bash
uvicorn main:app --reload --port 8000
```
Open **http://localhost:8000/docs** — FastAPI auto-generates an interactive page where you
can test both endpoints directly in the browser. Great for a live review demo.

## Endpoints

**`GET /health`** — confirms both models are loaded.

**`POST /predict/fraud`** — score a transaction:
```json
{
  "merchant": "fraud_Kirlin and Sons",
  "category": "shopping_net",
  "amt": 249.99,
  "gender": "M",
  "lat": 33.9659,
  "long": -80.9355,
  "city_pop": 333497,
  "merch_lat": 33.98,
  "merch_long": -81.20,
  "trans_date_trans_time": "2026-07-24 02:35:00",
  "dob": "1985-03-19"
}
```
Returns: `{"is_fraud": false, "fraud_probability": 0.0148, "threshold_used": 0.45}`

**`POST /predict/risk`** — score a borrower:
```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.85,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 2,
  "DebtRatio": 0.9,
  "MonthlyIncome": 3000,
  "NumberOfOpenCreditLinesAndLoans": 6,
  "NumberOfTimes90DaysLate": 1,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```
Returns: `{"high_risk": true, "delinquency_probability": 0.897, "threshold_used": 0.5}`

## Running this from Google Colab (for your review demo)

Colab doesn't expose ports directly to the internet, so use `pyngrok` to get a public URL:

```python
!pip install fastapi uvicorn pyngrok joblib nest_asyncio -q

import nest_asyncio, uvicorn
from pyngrok import ngrok

nest_asyncio.apply()
public_url = ngrok.connect(8000)
print("Public API URL:", public_url)

uvicorn.run(app, port=8000)  # 'app' imported from main.py
```
You'll need a free ngrok account/token for this. This is exactly the setup you'd use to
demo "real-time bank application integration" live in your review — a panel member could
hit your public URL from their own browser and get a live prediction.

## Next step: the dashboard

This API is what your dashboard (Streamlit, for example) should call instead of loading
the models directly — that keeps the ML logic in one place and the dashboard purely
about visualization. Ask for the Streamlit dashboard next and it'll be wired to call
these same two endpoints.
