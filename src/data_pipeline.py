"""Data loading and preprocessing pipeline"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_path="data/raw/telco_churn.csv"):
    """Load IBM Telco Churn dataset"""
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df):
    """Clean and encode features"""
    df = df.copy()

    # Fix TotalCharges (stored as text with blanks)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Churn']

    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes

    return df


def engineer_features(df):
    """Create domain-driven features"""
    df = df.copy()

    # Honeymoon period (tenure < 6 months)
    df['is_honeymoon_period'] = (df['tenure'] < 6).astype(int)

    # Month-to-month contract risk
    df['is_month_to_month'] = (df['Contract'] == 0).astype(int)

    # Price acceptance proxy
    df['monthly_charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # Service bundle depth
    service_cols = [col for col in df.columns if 'Service' in col or 'Phone' in col or 'Internet' in col]
    df['service_bundle_depth'] = df[service_cols].sum(axis=1)

    return df


def get_feature_names(df):
    """Get list of feature names (excluding target)"""
    return [col for col in df.columns if col != 'Churn']
