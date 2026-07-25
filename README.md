# 🏦 Enterprise Credit Risk & Default Decision Engine
# 🏦 Enterprise Credit Risk & Default Decision Engine

[![Live App](https://img.shields.io/badge/Streamlit-Live_Demo-ff4b4b.svg)](https://vandana-006-credit-risk-default-engine-appmain-qizlyh.streamlit.app/)

An end-to-end Machine Learning pipeline and interactive decisioning engine that predicts Probability of Default ($PD$) and calculates Expected Loss ($EL$) for retail credit applicants.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-0.94_AUC-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

## 📌 Project Overview
Credit risk modeling is central to banking stability. This project moves beyond standard classification metrics by focusing on **calibrated default probabilities**, financial feature engineering, and **regulatory explainability** via SHAP value attribution.

### Key Financial Features Engineered:
- **Loan-to-Income Ratio:** Measures applicant leverage relative to earnings.
- **Annual Interest Burden:** Calculates exact annual dollar cost of debt service.
- **Interest Coverage Ratio:** Assesses financial buffer against default.

---

## 📊 Model Performance

| Metric | Score | Financial Impact |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.9399** | Exceptional separation between default and non-default profiles |
| **Precision (Default)** | **0.89** | Low false-alarm rate on non-defaulters |
| **Recall (Default)** | **0.76** | Captures 76% of high-risk defaults prior to issuance |

---

## 🛠️ How to Run Locally

1. **Clone Repository & Set Up Virtual Environment:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/credit-risk-default-engine.git](https://github.com/YOUR-USERNAME/credit-risk-default-engine.git)
   cd credit-risk-default-engine
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt