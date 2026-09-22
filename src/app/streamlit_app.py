"""
FinSphere AI — Enterprise Intelligence Web Application
Full interactive suite for Credit Card Fraud Detection, Credit Risk Underwriting,
Explainable AI (SHAP Waterfall), Batch CSV Intelligence, and Model Governance & Drift.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.fraud_pipeline import FraudDetectionPipeline
from src.ml.risk_pipeline import CreditRiskPipeline
from src.ml.metrics_utils import calculate_psi

st.set_page_config(
    page_title="FinSphere AI — Financial Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 12px;
    }
    .badge-safe { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; }
    .badge-warn { background-color: #FEF08A; color: #854D0E; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; }
    .badge-danger { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    fraud_pipe, risk_pipe = None, None
    f_paths = ["models/fraud_model.pkl", "API-layer/models/fraud_model.pkl"]
    r_paths = ["models/risk_model.pkl", "API-layer/models/risk_model.pkl"]

    for p in f_paths:
        if os.path.exists(p):
            try:
                fraud_pipe = FraudDetectionPipeline.load(p)
                break
            except Exception as e:
                pass

    for p in r_paths:
        if os.path.exists(p):
            try:
                risk_pipe = CreditRiskPipeline.load(p)
                break
            except Exception as e:
                pass

    return fraud_pipe, risk_pipe


fraud_pipeline, risk_pipeline = load_models()

# Sidebar
st.sidebar.title("⚡ FinSphere AI")
st.sidebar.markdown("**Enterprise Financial Risk & Fraud Intelligence**")
nav = st.sidebar.radio(
    "Modules",
    [
        "🛡️ Real-Time Fraud Detection",
        "📊 Credit Risk Underwriting",
        "📁 Batch CSV Intelligence",
        "📈 Model Benchmarks & ROI Matrix",
        "🔍 Data Drift & PSI Monitor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🟢 System Status")
st.sidebar.markdown(f"- Fraud Model: {'`Ready (XGBoost)`' if fraud_pipeline else '`Not Loaded`'}")
st.sidebar.markdown(f"- Risk Model: {'`Ready (XGBoost)`' if risk_pipeline else '`Not Loaded`'}")
st.sidebar.markdown("---")
st.sidebar.caption("FinSphere AI v2.0 • End-to-End Enterprise ML Pipeline")

# -----------------------------------------------------------------------------
# 1. Real-Time Fraud Detection
# -----------------------------------------------------------------------------
if nav == "🛡️ Real-Time Fraud Detection":
    st.markdown('<div class="main-title">🛡️ Real-Time Fraud Detection & SHAP Explanation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Live transaction risk scoring powered by XGBoost with sub-15ms latency and local TreeSHAP explainability.</div>', unsafe_allow_html=True)

    if not fraud_pipeline:
        st.warning("⚠️ Fraud model bundle not found. Please run `python src/ml/train_all.py` to train and export the models.")
    else:
        # Preset selector
        preset = st.selectbox(
            "Load Quick Test Scenario:",
            ["Custom Input", "Scenario A: Normal Grocery Store Purchase", "Scenario B: Suspicious 2 AM High-Value Online Transaction", "Scenario C: High-Distance Travel Transaction"]
        )

        # Defaults
        merchant_def, cat_def, amt_def, gender_def = "fraud_Kirlin and Sons", "shopping_net", 249.99, "M"
        lat_def, long_def, m_lat_def, m_long_def = 33.9659, -80.9355, 33.98, -81.20
        city_pop_def, trans_time_def, dob_def = 333497, "2026-07-24 14:35:00", "1985-03-19"

        if preset == "Scenario A: Normal Grocery Store Purchase":
            merchant_def, cat_def, amt_def, trans_time_def = "fraud_Rippin_Carrington", "grocery_pos", 34.50, "2026-07-24 11:15:00"
            m_lat_def, m_long_def = 33.97, -80.94
        elif preset == "Scenario B: Suspicious 2 AM High-Value Online Transaction":
            merchant_def, cat_def, amt_def, trans_time_def = "fraud_Kirlin and Sons", "shopping_net", 899.00, "2026-07-24 02:45:00"
            m_lat_def, m_long_def = 38.50, -120.30
        elif preset == "Scenario C: High-Distance Travel Transaction":
            merchant_def, cat_def, amt_def, trans_time_def = "fraud_Cruickshank_Shields", "travel", 640.00, "2026-07-24 23:10:00"
            m_lat_def, m_long_def = 40.71, -74.00

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Transaction Parameters")
            c_a, c_b = st.columns(2)
            with c_a:
                merchant = st.text_input("Merchant Name", merchant_def)
                category = st.selectbox("Category", [
                    "shopping_net", "grocery_pos", "misc_net", "travel", "gas_transport",
                    "shopping_pos", "entertainment", "personal_care", "food_dining", "health_fitness"
                ], index=0 if cat_def == "shopping_net" else 1)
                amt = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=50000.0, value=float(amt_def), step=10.0)
                gender = st.selectbox("Cardholder Gender", ["M", "F"], index=0 if gender_def == "M" else 1)
                city_pop = st.number_input("City Population", min_value=50, max_value=10000000, value=int(city_pop_def))
            with c_b:
                trans_date_trans_time = st.text_input("Transaction Timestamp", trans_time_def)
                dob = st.text_input("Cardholder Date of Birth", dob_def)
                lat = st.number_input("Cardholder Latitude", value=float(lat_def), format="%.4f")
                long = st.number_input("Cardholder Longitude", value=float(long_def), format="%.4f")
                merch_lat = st.number_input("Merchant Latitude", value=float(m_lat_def), format="%.4f")
                merch_long = st.number_input("Merchant Longitude", value=float(m_long_def), format="%.4f")

            custom_threshold = st.slider(
                "Decision Threshold", min_value=0.05, max_value=0.95,
                value=float(fraud_pipeline.best_threshold), step=0.05,
                help="Adjust sensitivity. Lower threshold increases fraud recall; higher threshold reduces false alerts."
            )

            score_btn = st.button("🚀 Score Transaction with SHAP Analysis", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Real-Time Scoring & Explainability")
            raw_input = {
                "merchant": merchant, "category": category, "amt": amt, "gender": gender,
                "lat": lat, "long": long, "city_pop": city_pop,
                "merch_lat": merch_lat, "merch_long": merch_long,
                "trans_date_trans_time": trans_date_trans_time, "dob": dob
            }

            if score_btn or preset != "Custom Input":
                t0 = time.perf_counter()
                pred = fraud_pipeline.predict(raw_input, threshold=custom_threshold)
                explanation = fraud_pipeline.explain(raw_input)
                latency = (time.perf_counter() - t0) * 1000.0

                prob = pred["fraud_probability"]
                is_fraud = pred["is_fraud"]

                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.1rem; font-weight:600;">Prediction Result:</span>
                        <span class="{ 'badge-danger' if is_fraud else 'badge-safe' }">
                            { '⚠️ FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE' }
                        </span>
                    </div>
                    <div style="margin-top:12px; font-size:2.2rem; font-weight:700; color:{ '#EF4444' if is_fraud else '#10B981' };">
                        {prob*100:.2f}% <span style="font-size:1rem; color:#64748B;">fraud probability</span>
                    </div>
                    <div style="font-size:0.85rem; color:#64748B; margin-top:6px;">
                        Decision Cutoff: <b>{custom_threshold:.2f}</b> • Inference Latency: <b>{latency:.2f} ms</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 🔍 Local TreeSHAP Feature Contributions")
                reasons = explanation["top_reasons"]
                feat_df = pd.DataFrame(reasons)

                fig, ax = plt.subplots(figsize=(6, 3.8))
                colors = ['#EF4444' if x > 0 else '#10B981' for x in feat_df['shap_value']]
                y_pos = np.arange(len(feat_df))
                ax.barh(y_pos, feat_df['shap_value'], color=colors, align='center', height=0.65)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feat_df['feature'], fontsize=9)
                ax.invert_yaxis()
                ax.axvline(0, color='#94A3B8', linestyle='--', linewidth=0.8)
                ax.set_xlabel('SHAP Impact on Log-Odds', fontsize=8.5)
                ax.set_title('Top Drivers Influencing this Prediction', fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

                st.dataframe(
                    feat_df[['feature', 'feature_value', 'impact', 'shap_value']],
                    use_container_width=True, hide_index=True
                )

# -----------------------------------------------------------------------------
# 2. Credit Risk Underwriting
# -----------------------------------------------------------------------------
elif nav == "📊 Credit Risk Underwriting":
    st.markdown('<div class="main-title">📊 Credit Risk & Regulatory Underwriting Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate borrower delinquency probability, risk tiers, and adverse action compliance reasons using TreeSHAP.</div>', unsafe_allow_html=True)

    if not risk_pipeline:
        st.warning("⚠️ Credit risk model bundle not found. Please run `python src/ml/train_all.py` to train and export.")
    else:
        preset_r = st.selectbox(
            "Load Quick Borrower Profile:",
            ["Custom Borrower", "Profile A: Prime Low-Risk Applicant", "Profile B: Moderate Debt Stress Applicant", "Profile C: Subprime / High Delinquency Risk"]
        )

        util_def, age_def, l30_def, debt_def, inc_def = 0.25, 45, 0, 0.35, 6500.0
        lines_def, l90_def, re_def, l60_def, dep_def = 8, 0, 1, 0, 1

        if preset_r == "Profile A: Prime Low-Risk Applicant":
            util_def, age_def, l30_def, debt_def, inc_def = 0.12, 52, 0, 0.22, 9500.0
            lines_def, l90_def, re_def, l60_def, dep_def = 10, 0, 2, 0, 2
        elif preset_r == "Profile B: Moderate Debt Stress Applicant":
            util_def, age_def, l30_def, debt_def, inc_def = 0.78, 38, 1, 0.65, 4200.0
            lines_def, l90_def, re_def, l60_def, dep_def = 5, 0, 1, 0, 1
        elif preset_r == "Profile C: Subprime / High Delinquency Risk":
            util_def, age_def, l30_def, debt_def, inc_def = 0.98, 29, 3, 0.88, 2500.0
            lines_def, l90_def, re_def, l60_def, dep_def = 4, 2, 0, 1, 0

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Borrower Financial Attributes")
            c_a, c_b = st.columns(2)
            with c_a:
                revolving_util = st.number_input("Revolving Utilization Ratio (0.0 - 2.0)", min_value=0.0, max_value=5.0, value=float(util_def), step=0.05)
                age = st.number_input("Borrower Age", min_value=18, max_value=100, value=int(age_def))
                monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, max_value=200000.0, value=float(inc_def), step=250.0)
                debt_ratio = st.number_input("Debt-to-Income Ratio (0.0 - 5.0)", min_value=0.0, max_value=10.0, value=float(debt_def), step=0.05)
                dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=int(dep_def))
            with c_b:
                open_lines = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=int(lines_def))
                real_estate_lines = st.number_input("Real Estate Loans/Lines", min_value=0, max_value=25, value=int(re_def))
                past_due_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=18, value=int(l30_def))
                past_due_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=18, value=int(l60_def))
                past_due_90 = st.number_input("Times 90+ Days Past Due", min_value=0, max_value=18, value=int(l90_def))

            risk_threshold = st.slider(
                "Risk Approval Threshold", min_value=0.10, max_value=0.90,
                value=float(risk_pipeline.best_threshold), step=0.05,
                help="Threshold for flagging an applicant as High Risk Delinquency."
            )

            underwrite_btn = st.button("📑 Underwrite Application & Generate Adverse Factors", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Underwriting Decision & Regulatory Explanations")
            borrower_raw = {
                "RevolvingUtilizationOfUnsecuredLines": revolving_util,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": past_due_90,
                "NumberRealEstateLoansOrLines": real_estate_lines,
                "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
                "NumberOfDependents": dependents
            }

            if underwrite_btn or preset_r != "Custom Borrower":
                t0 = time.perf_counter()
                pred_r = risk_pipeline.predict(borrower_raw, threshold=risk_threshold)
                explanation_r = risk_pipeline.explain(borrower_raw)
                latency_r = (time.perf_counter() - t0) * 1000.0

                prob_r = pred_r["delinquency_probability"]
                tier = pred_r["risk_tier"]

                badge_class = "badge-safe" if tier == "LOW_RISK_APPROVE" else ("badge-warn" if tier == "MODERATE_RISK_REVIEW" else "badge-danger")

                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.1rem; font-weight:600;">Risk Classification:</span>
                        <span class="{badge_class}">{tier}</span>
                    </div>
                    <div style="margin-top:12px; font-size:2.2rem; font-weight:700; color:{ '#EF4444' if prob_r >= risk_threshold else '#10B981' };">
                        {prob_r*100:.2f}% <span style="font-size:1rem; color:#64748B;">delinquency risk</span>
                    </div>
                    <div style="font-size:0.9rem; margin-top:8px; color:#1E293B;">
                        <b>Recommended Action:</b> {pred_r['recommended_action']}
                    </div>
                    <div style="font-size:0.85rem; color:#64748B; margin-top:4px;">
                        Approval Cutoff: <b>{risk_threshold:.2f}</b> • Latency: <b>{latency_r:.2f} ms</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 📜 Adverse Action / Top Risk Drivers (ECOA Compliant)")
                reasons_r = explanation_r["top_risk_factors"]
                feat_r_df = pd.DataFrame(reasons_r)

                fig_r, ax_r = plt.subplots(figsize=(6, 3.4))
                colors_r = ['#EF4444' if x > 0 else '#10B981' for x in feat_r_df['shap_value']]
                y_pos_r = np.arange(len(feat_r_df))
                ax_r.barh(y_pos_r, feat_r_df['shap_value'], color=colors_r, align='center', height=0.65)
                ax_r.set_yticks(y_pos_r)
                ax_r.set_yticklabels(feat_r_df['feature'], fontsize=9)
                ax_r.invert_yaxis()
                ax_r.axvline(0, color='#94A3B8', linestyle='--', linewidth=0.8)
                ax_r.set_xlabel('SHAP Impact on Delinquency Log-Odds', fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig_r)

                st.dataframe(
                    feat_r_df[['feature', 'feature_value', 'impact', 'shap_value']],
                    use_container_width=True, hide_index=True
                )

# -----------------------------------------------------------------------------
# 3. Batch CSV Intelligence
# -----------------------------------------------------------------------------
elif nav == "📁 Batch CSV Intelligence":
    st.markdown('<div class="main-title">📁 High-Throughput Batch CSV Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload bulk transactions or loan applicants to score thousands of records in seconds with downloadable analytics.</div>', unsafe_allow_html=True)

    task_type = st.radio("Select Domain:", ["Fraud Detection Batch", "Credit Risk Batch"], horizontal=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df_upload):,} rows** with columns:", list(df_upload.columns))

        if st.button("⚡ Run Batch Prediction Pipeline", type="primary"):
            with st.spinner("Processing records..."):
                t0 = time.perf_counter()
                if task_type == "Fraud Detection Batch":
                    if not fraud_pipeline:
                        st.error("Fraud model not loaded.")
                    else:
                        probs = []
                        is_frauds = []
                        for _, row in df_upload.iterrows():
                            res = fraud_pipeline.predict(row.to_dict())
                            probs.append(res["fraud_probability"])
                            is_frauds.append(res["is_fraud"])
                        df_upload["fraud_probability"] = probs
                        df_upload["predicted_is_fraud"] = is_frauds
                else:
                    if not risk_pipeline:
                        st.error("Risk model not loaded.")
                    else:
                        probs = []
                        tiers = []
                        for _, row in df_upload.iterrows():
                            res = risk_pipeline.predict(row.to_dict())
                            probs.append(res["delinquency_probability"])
                            tiers.append(res["risk_tier"])
                        df_upload["delinquency_probability"] = probs
                        df_upload["risk_tier"] = tiers

                duration = time.perf_counter() - t0
                st.success(f"Processed {len(df_upload):,} records in {duration:.2f} seconds ({len(df_upload)/max(duration, 0.001):.0f} records/sec)!")

                st.dataframe(df_upload.head(50), use_container_width=True)

                csv_data = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Scored CSV Report",
                    csv_data,
                    "finsphere_scored_batch.csv",
                    "text/csv",
                    key="download-csv"
                )
    else:
        st.info("💡 Tip: You can test batch upload by uploading a small slice of `fraudTest.csv` or `cs-training.csv`.")

# -----------------------------------------------------------------------------
# 4. Model Benchmarks & ROI Matrix
# -----------------------------------------------------------------------------
elif nav == "📈 Model Benchmarks & ROI Matrix":
    st.markdown('<div class="main-title">📈 Model Performance Benchmarks & Business Loss Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical validation: Comparing baseline Random Forest vs. Optimized XGBoost with feature engineering.</div>', unsafe_allow_html=True)

    summary_file = "models/benchmark_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            benchmarks = json.load(f)

        tab1, tab2 = st.tabs(["🛡️ Fraud Detection Benchmarks", "📊 Credit Risk Benchmarks"])

        with tab1:
            f_data = benchmarks.get("fraud_detection", {})
            if f_data:
                rf = f_data.get("rf_baseline", {})
                xgb = f_data.get("xgboost_optimal", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("XGBoost F1-Score", f"{xgb.get('f1', 0):.4f}", f"{(xgb.get('f1', 0) - rf.get('f1', 0)):+.4f} vs Baseline")
                c2.metric("XGBoost PR-AUC", f"{xgb.get('pr_auc', 0):.4f}", f"{(xgb.get('pr_auc', 0) - rf.get('pr_auc', 0)):+.4f} vs Baseline")
                c3.metric("XGBoost ROC-AUC", f"{xgb.get('roc_auc', 0):.4f}", f"{(xgb.get('roc_auc', 0) - rf.get('roc_auc', 0)):+.4f} vs Baseline")
                c4.metric("Optimal Threshold", f"{f_data.get('best_threshold', 0.45):.3f}")

                st.markdown("#### Quantitative Benchmark Table")
                comparison_df = pd.DataFrame([
                    {"Model": "Random Forest Baseline", "Accuracy": f"{rf.get('accuracy', 0):.4f}", "Precision": f"{rf.get('precision', 0):.4f}", "Recall": f"{rf.get('recall', 0):.4f}", "F1-Score": f"{rf.get('f1', 0):.4f}", "PR-AUC": f"{rf.get('pr_auc', 0):.4f}", "ROC-AUC": f"{rf.get('roc_auc', 0):.4f}"},
                    {"Model": "Optimized XGBoost (Proposed)", "Accuracy": f"{xgb.get('accuracy', 0):.4f}", "Precision": f"{xgb.get('precision', 0):.4f}", "Recall": f"{xgb.get('recall', 0):.4f}", "F1-Score": f"{xgb.get('f1', 0):.4f}", "PR-AUC": f"{xgb.get('pr_auc', 0):.4f}", "ROC-AUC": f"{xgb.get('roc_auc', 0):.4f}"}
                ])
                st.table(comparison_df)

                # Cost Curve
                cost_curve = f_data.get("cost_curve", [])
                if cost_curve:
                    st.markdown("#### 💰 Financial Expected Loss vs. Decision Cutoff")
                    cc_df = pd.DataFrame(cost_curve)
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.plot(cc_df['threshold'], cc_df['cost'], color='#EF4444', linewidth=2.5, marker='o', label='Expected Financial Loss ($)')
                    ax.axvline(f_data.get('best_cost_threshold', 0.45), color='#10B981', linestyle='--', label=f"Cost-Optimal Threshold ({f_data.get('best_cost_threshold', 0.45)})")
                    ax.set_xlabel('Probability Decision Threshold')
                    ax.set_ylabel('Total Dollar Loss ($)')
                    ax.legend()
                    st.pyplot(fig)

        with tab2:
            r_data = benchmarks.get("credit_risk", {})
            if r_data:
                rf_r = r_data.get("rf_baseline", {})
                xgb_r = r_data.get("xgboost_optimal", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("XGBoost F1-Score", f"{xgb_r.get('f1', 0):.4f}", f"{(xgb_r.get('f1', 0) - rf_r.get('f1', 0)):+.4f} vs Baseline")
                c2.metric("XGBoost PR-AUC", f"{xgb_r.get('pr_auc', 0):.4f}", f"{(xgb_r.get('pr_auc', 0) - rf_r.get('pr_auc', 0)):+.4f} vs Baseline")
                c3.metric("XGBoost ROC-AUC", f"{xgb_r.get('roc_auc', 0):.4f}", f"{(xgb_r.get('roc_auc', 0) - rf_r.get('roc_auc', 0)):+.4f} vs Baseline")
                c4.metric("Optimal Threshold", f"{r_data.get('best_threshold', 0.35):.3f}")

                st.markdown("#### Quantitative Benchmark Table")
                comparison_r_df = pd.DataFrame([
                    {"Model": "Random Forest Baseline", "Accuracy": f"{rf_r.get('accuracy', 0):.4f}", "Precision": f"{rf_r.get('precision', 0):.4f}", "Recall": f"{rf_r.get('recall', 0):.4f}", "F1-Score": f"{rf_r.get('f1', 0):.4f}", "PR-AUC": f"{rf_r.get('pr_auc', 0):.4f}", "ROC-AUC": f"{rf_r.get('roc_auc', 0):.4f}"},
                    {"Model": "Optimized XGBoost (Proposed)", "Accuracy": f"{xgb_r.get('accuracy', 0):.4f}", "Precision": f"{xgb_r.get('precision', 0):.4f}", "Recall": f"{xgb_r.get('recall', 0):.4f}", "F1-Score": f"{xgb_r.get('f1', 0):.4f}", "PR-AUC": f"{xgb_r.get('pr_auc', 0):.4f}", "ROC-AUC": f"{xgb_r.get('roc_auc', 0):.4f}"}
                ])
                st.table(comparison_r_df)
    else:
        st.info("Run `python src/ml/train_all.py` to generate the complete empirical benchmark summary.")

# -----------------------------------------------------------------------------
# 5. Data Drift & PSI Monitor
# -----------------------------------------------------------------------------
elif nav == "🔍 Data Drift & PSI Monitor":
    st.markdown('<div class="main-title">🔍 Data Drift & Population Stability Index (PSI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enterprise ML Governance: Monitor covariate shift and distribution divergence between training baseline and incoming production data.</div>', unsafe_allow_html=True)

    np.random.seed(42)
    # Simulated baseline and shifted production streams
    feature_selected = st.selectbox("Select Feature to Inspect:", ["Transaction Amount ($)", "Revolving Credit Utilization", "Customer Age"])
    shift_severity = st.slider("Simulate Production Covariate Shift Severity:", min_value=0.0, max_value=2.0, value=0.2, step=0.1)

    if feature_selected == "Transaction Amount ($)":
        baseline = np.random.exponential(scale=65.0, size=2000)
        production = np.random.exponential(scale=65.0 * (1.0 + shift_severity), size=1500) + (np.random.normal(20 * shift_severity, 5, 1500))
    elif feature_selected == "Revolving Credit Utilization":
        baseline = np.random.beta(a=2, b=5, size=2000) * 1.5
        production = np.random.beta(a=2 + shift_severity, b=5 - min(shift_severity, 2.0), size=1500) * 1.5
    else:
        baseline = np.random.normal(loc=48, scale=12, size=2000).clip(18, 90)
        production = np.random.normal(loc=48 - (8 * shift_severity), scale=12, size=1500).clip(18, 90)

    psi_score, buckets = calculate_psi(baseline, production, num_buckets=10)

    if psi_score < 0.10:
        status = "🟢 STABLE (PSI < 0.10) — No significant distribution shift"
    elif psi_score < 0.25:
        status = "🟡 MODERATE DRIFT (0.10 ≤ PSI < 0.25) — Warning: Monitor closely"
    else:
        status = "🔴 SIGNIFICANT DRIFT (PSI ≥ 0.25) — Trigger automated retraining pipeline!"

    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.1rem; font-weight:600;">Population Stability Index (PSI):</div>
        <div style="font-size:2rem; font-weight:700; color:{ '#10B981' if psi_score < 0.1 else ('#F59E0B' if psi_score < 0.25 else '#EF4444') };">
            {psi_score:.4f}
        </div>
        <div style="font-size:0.95rem; font-weight:600; margin-top:6px;">{status}</div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.kdeplot(baseline, label="Training Baseline", color="#3B82F6", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(production, label="Current Production Stream", color="#EF4444", fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"Distribution Comparison: {feature_selected}")
    ax.legend()
    st.pyplot(fig)

    st.markdown("##### 📊 Bucket-Level PSI Decomposition")
    st.dataframe(pd.DataFrame(buckets), use_container_width=True, hide_index=True)
