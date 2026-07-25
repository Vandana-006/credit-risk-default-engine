import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from data_loader import load_and_clean_data

def train_credit_model():
    # 1. Load Cleaned Data
    df = load_and_clean_data()
    
    # 2. Separate Features (X) and Target (y)
    # Target column is 'loan_status' (1 = Default, 0 = Non-Default)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Handle Class Imbalance (More non-defaulters than defaulters)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print("\nTraining Monotone XGBoost Model...")
    base_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 5. Calibrate Probabilities (Essential for Financial Risk Engine)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # 6. Evaluate Model Performance
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nModel AUC-ROC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save Model & Feature List for Dashboard Production
    joblib.dump(calibrated_model, "src/credit_model.pkl")
    joblib.dump(list(X.columns), "src/model_features.pkl")
    print("Model and features successfully saved to `src/credit_model.pkl` & `src/model_features.pkl`!")

if __name__ == "__main__":
    train_credit_model()