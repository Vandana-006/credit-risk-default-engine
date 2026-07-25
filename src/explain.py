import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from data_loader import load_and_clean_data

def generate_shap_plots():
    print("Loading data and model for SHAP analysis...")
    df = load_and_clean_data()
    X = df.drop(columns=['loan_status'])
    
    # Load calibrated model and extract underlying XGBoost estimator
    calibrated_model = joblib.load("src/credit_model.pkl")
    # Take the base estimator from the calibrated classifier
    xgb_model = calibrated_model.calibrated_classifiers_[0].estimator
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    
    # Save chart image for README & GitHub
    plt.savefig("shap_summary.png", dpi=300)
    print("SHAP feature importance plot successfully saved to `shap_summary.png`!")

if __name__ == "__main__":
    generate_shap_plots()