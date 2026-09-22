# FinSphere AI — Resume Impact Guide

> Use this document to craft your resume bullet points, LinkedIn descriptions, and interview talking points.
> All metrics are **empirically verified** results from running `python src/ml/train_all.py` on the actual datasets.

---

## Verified Model Benchmark Numbers (Source: `models/benchmark_summary.json`)

### Fraud Detection (1,293,485 credit card transactions)
| Metric | Baseline (Random Forest) | Improved (XGBoost + Feature Engineering) | Delta |
|---|---|---|---|
| F1-Score (Fraud class) | 0.7759 | **0.8132** | +3.7 pp |
| PR-AUC | 0.8274 | **0.8781** | +5.1 pp |
| ROC-AUC | 0.9953 | **0.9974** | +0.2 pp |
| Decision Threshold | 0.50 (default) | **0.825** (cost-optimal) | Tuned |

### Credit Risk Assessment (150,000 borrowers)
| Metric | Baseline (Random Forest) | Improved (XGBoost + Stress Features) | Delta |
|---|---|---|---|
| F1-Score (Delinquent class) | 0.4352 | **0.4497** | +1.5 pp |
| PR-AUC | 0.3743 | **0.3987** | +2.4 pp |
| ROC-AUC | 0.8619 | **0.8686** | +0.7 pp |

---

## Ready-to-Use Resume Bullet Points

### For Resume (Action Verb + Number + Result Format)

**Fraud Detection Model:**
- Engineered a production-grade XGBoost credit card fraud detection system on 1.29M transactions (0.57% fraud rate), achieving **F1-score of 0.813** and **PR-AUC of 0.878** — a +3.7pp and +5.1pp improvement over the Random Forest baseline
- Designed cyclical temporal encodings (sin/cos hour), Haversine geodesic customer-merchant distance features, and category-level amount z-score anomaly signals, improving minority-class recall for an extreme class imbalance problem
- Implemented financial cost-matrix threshold optimization, reducing expected dollar loss by shifting the decision boundary from 0.50 to 0.825, minimizing false negative costs (missed fraud, \$150/incident) vs. false positive friction (\$15/incident)
- Integrated real-time **TreeSHAP explainability** generating local feature attributions per prediction (<15ms latency), enabling transparent model audit trails via the REST API's `/explain/fraud` endpoint

**Credit Risk Model:**
- Developed a credit delinquency prediction system achieving **ROC-AUC of 0.869** on 150K borrowers with 6.7% delinquency rate, surpassing Random Forest baseline by 0.7pp
- Constructed novel financial stress features including weighted delinquency severity scores `(1×30d + 2×60d + 3.5×90d)`, disposable income estimates, and revolving utilization stress indices — reflecting credit bureau severity tiers and regulatory lending standards
- Generated regulatory ECOA-compliant adverse action notices using TreeSHAP values, providing quantitative factor-level reasons for credit denial decisions
- Applied XGBoost `scale_pos_weight=13.96` for cost-sensitive learning on the imbalanced credit dataset, optimizing for delinquent-class recall without sacrificing portfolio precision

**Full-Stack ML Production System:**
- Deployed dual-model prediction service using **FastAPI** with 8 REST endpoints (single prediction, SHAP explanation, batch CSV scoring, live model metrics), achieving sub-10ms inference latency per transaction
- Built an interactive **Streamlit web application** featuring real-time risk gauges, dynamic SHAP waterfall charts, batch CSV upload/download, ROC/PR curve comparison dashboards, and Population Stability Index (PSI) drift monitoring
- Containerized the full ML pipeline using **Docker**, enabling reproducible training environments and one-command deployment
- Wrote a comprehensive **pytest suite (11 tests)** covering feature engineering pipelines, metrics utilities, PSI drift calculations, and all API endpoints

---

## LinkedIn About / Project Description

**FinSphere AI — Enterprise Financial Risk Intelligence Platform** (Personal ML Engineering Project)

Built a production-grade dual-model AI system for real-time fraud detection and credit risk underwriting using XGBoost, SHAP explainability, FastAPI, and Streamlit.

**Key Technical Achievements:**
- **Fraud Detection**: F1-Score 0.813 / PR-AUC 0.878 on 1.29M transactions with 0.57% fraud rate using advanced feature engineering (cyclical time, geodesic distance, category deviation z-scores)
- **Credit Risk**: ROC-AUC 0.869 with regulatory-compliant adverse action notices via TreeSHAP
- **Production API**: 8 FastAPI endpoints with sub-10ms inference latency, SHAP local explanations, and batch scoring
- **Interactive Dashboard**: Real-time Streamlit UI with SHAP waterfall charts, PSI drift monitoring, and ROI cost-benefit calculator

---

## Interview Talking Points

### Q: "Tell me about a challenging ML engineering problem you solved."

**STAR Answer:**
- **Situation**: Credit card fraud detection on a highly imbalanced dataset (0.57% fraud rate — 99.4% accuracy from a dummy classifier that predicts all legitimate).
- **Task**: Build a production-grade model that maximizes real fraud capture while minimizing false decline friction for legitimate customers.
- **Action**: (1) Replaced accuracy with PR-AUC as the primary metric for imbalanced regimes. (2) Applied `scale_pos_weight=175` (neg/pos ratio) in XGBoost for cost-sensitive learning. (3) Engineered Haversine geodesic distance and cyclical time sin/cos features. (4) Ran financial cost-matrix threshold search optimizing dollar loss `(FP×$15 + FN×$150)` to find the optimal 0.825 cutoff. (5) Added TreeSHAP for per-transaction explainability.
- **Result**: PR-AUC of 0.878 (+5.1pp over baseline) and F1-Score of 0.813 (+3.7pp), deployed behind a FastAPI service with <10ms latency.

---

### Q: "How did you handle class imbalance?"

**Four-layer approach:**
1. **Algorithm**: XGBoost `scale_pos_weight` = (negatives / positives) — directly penalizes misclassifying the minority class proportionally to imbalance ratio
2. **Threshold Tuning**: Scanned thresholds 0.05–0.95 in 0.025 increments, optimizing for F1 AND for a financial expected-cost function (FP cost vs. FN cost) — not just defaulting to 0.5
3. **Evaluation Metric**: Used PR-AUC (Precision-Recall Area Under Curve), not ROC-AUC alone — PR-AUC is more sensitive to minority class performance in severe imbalance settings
4. **Reporting**: Stratified train/test split preserving class proportions for unbiased evaluation

---

### Q: "How do you explain ML predictions to non-technical stakeholders?"

**SHAP Explainability:**
- Used **TreeSHAP** (fast, exact, model-specific SHAP implementation for tree ensembles) — not approximations
- Each prediction returns a list of feature contributions: "The transaction amount was 3.2σ above the average for this merchant category (+0.42 fraud log-odds), and the merchant is 285km from the customer's home address (+0.31 fraud log-odds)"
- For credit risk, this directly maps to ECOA-compliant adverse action notices: "Primary reasons for decline: (1) Revolving credit utilization 98%, (2) 3 instances of 30–59 day past due in past 2 years"
- Visualized as interactive SHAP waterfall charts in the Streamlit dashboard for analyst review

---

### Q: "What is PSI and why does it matter in production ML?"

**Answer:**
- PSI (Population Stability Index) measures distribution shift between training data and production data
- PSI < 0.10: Distribution is stable — no action needed
- 0.10 ≤ PSI < 0.25: Moderate drift — monitor closely
- PSI ≥ 0.25: Significant covariate shift — trigger retraining pipeline
- I implemented PSI calculation over 10 quantile buckets: `PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected)`
- This is the industry standard at major financial institutions for model governance and MLOps drift alerts

---

## Skills Demonstrated by This Project

| Domain | Specific Skills |
|---|---|
| ML Engineering | XGBoost, scikit-learn, imbalanced learning, feature engineering, hyperparameter optimization |
| Explainability (XAI) | TreeSHAP, local feature attribution, global feature importance |
| MLOps & Governance | Threshold optimization, PSI drift monitoring, serialized model bundles, benchmark logging |
| Production APIs | FastAPI, Pydantic v2, REST design, batch scoring, async endpoints |
| Web Applications | Streamlit, interactive dashboards, real-time visualization |
| Software Engineering | pytest, Docker, modular package design, type annotations |
| Finance Domain | Credit risk, fraud detection, regulatory compliance (ECOA adverse action) |

---

## GitHub Repository Checklist

- [x] Clean src/ package structure (ml/, api/, app/)
- [x] Comprehensive README.md with benchmark tables & usage examples
- [x] Working Jupyter notebooks with full markdown explanations
- [x] Dockerfile for reproducible deployment
- [x] requirements.txt with pinned versions
- [x] pytest test suite (11 tests, all passing)
- [x] .gitignore excluding large CSVs and model pkl files
- [x] models/benchmark_summary.json with verified quantitative metrics
