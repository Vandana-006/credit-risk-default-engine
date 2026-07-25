import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Credit Risk & Default Decision Engine",
    page_icon="🏦",
    layout="wide"
)

# Load Trained Model and Feature Names
@st.cache_resource
def load_assets():
    model = joblib.load("src/credit_model.pkl")
    features = joblib.load("src/model_features.pkl")
    return model, features

model, model_features = load_assets()

st.title("🏦 Enterprise Credit Risk & Default Decision Engine")
st.markdown("Assess loan applicant default risk, Expected Loss ($EL$), and credit decisioning using calibrated Machine Learning models.")

st.sidebar.header("📋 Applicant Profile Input")

# --- Interactive Sidebar Inputs ---
person_age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=100, value=28)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=55000, step=1000)
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)

loan_amnt = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=12000, step=500)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.5, step=0.1)

person_home_ownership = st.sidebar.selectbox("Home Ownership Status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Assigned Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.sidebar.radio("Prior Default on File?", ["No", "Yes"])

# --- Compute Features ---
annual_interest_burden = loan_amnt * (loan_int_rate / 100.0)
loan_to_income_ratio = loan_amnt / (person_income + 1)
interest_coverage = person_income / (annual_interest_burden + 1)

# Construct Input Dataframe
raw_input = {
    'person_age': person_age,
    'person_income': person_income,
    'person_emp_length': person_emp_length,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_to_income_ratio, # Aligning proxy column
    'cb_person_cred_hist_length': 3,            # Default proxy length
    'loan_to_income_ratio': loan_to_income_ratio,
    'annual_interest_burden': annual_interest_burden,
    'interest_coverage': interest_coverage,
    f'person_home_ownership_{person_home_ownership}': True,
    f'loan_intent_{loan_intent}': True,
    f'loan_grade_{loan_grade}': True,
    'cb_person_default_on_file_Y': True if cb_person_default_on_file == "Yes" else False
}

# Create a DataFrame matching exact model features
input_df = pd.DataFrame([raw_input])
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = False

input_df = input_df[model_features]

# --- Model Inference ---
if st.button("🚀 Calculate Credit Risk & Decision", type="primary"):
    default_prob = model.predict_proba(input_df)[0][1]
    
    # Financial Expected Loss (EL) Calculation: EL = PD * LGD * EAD
    # Standard Loss Given Default (LGD) assumption = 45%
    lgd = 0.45
    expected_loss = default_prob * lgd * loan_amnt

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Probability of Default (PD)", f"{default_prob:.1%}")
    with col2:
        st.metric("Expected Loss ($EL$)", f"${expected_loss:,.2f}")
    with col3:
        if default_prob < 0.20:
            st.success("Decision: APPROVED ✅ (Low Risk)")
        elif default_prob < 0.45:
            st.warning("Decision: MANUAL REVIEW ⚠️ (Medium Risk)")
        else:
            st.error("Decision: REJECTED ❌ (High Risk)")

    st.markdown("---")
    st.subheader("📊 Financial Exposure Summary")
    st.json({
        "Loan Amount Requested": f"${loan_amnt:,.2f}",
        "Annual Interest Burden": f"${annual_interest_burden:,.2f}",
        "Debt-to-Income Ratio": f"{loan_to_income_ratio:.2%}",
        "Interest Coverage Ratio": f"{interest_coverage:.2f}x"
    })