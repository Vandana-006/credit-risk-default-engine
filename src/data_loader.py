import pandas as pd
import numpy as np

def load_and_clean_data(file_path="data/credit_risk_dataset.csv"):
    """
    Loads raw credit data, removes unrealistic outliers, handles missing values,
    engineers financial features, and encodes categorical columns.
    """
    print("Loading raw credit risk data...")
    df = pd.read_csv(file_path)
    
    # 1. Clean Unrealistic Outliers (Domain Rule)
    # Age over 100 or work experience over 60 years are almost certainly entry errors
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 60]
    
    # 2. Financial Feature Engineering
    # Feature A: Debt-to-Income Proxy
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
    
    # Feature B: Annual Interest Burden ($ amount of interest per year)
    df['annual_interest_burden'] = df['loan_amnt'] * (df['loan_int_rate'] / 100.0)
    
    # Feature C: Interest Coverage Ratio
    df['interest_coverage'] = df['person_income'] / (df['annual_interest_burden'] + 1)

    # 3. Handle Missing Values
    # Impute employment length with median
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    
    # Impute missing interest rates with median interest rate of that specific loan grade
    df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
        lambda x: x.fillna(x.median())
    )

    # 4. One-Hot Encoding Categorical Features
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Data pipeline complete! Dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    data = load_and_clean_data()
    print("\nFirst 2 rows of cleaned data:")
    print(data.head(2))