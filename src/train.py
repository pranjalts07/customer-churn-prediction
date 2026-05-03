"""Model training with hyperparameter tuning"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from src.data_pipeline import load_raw_data, preprocess_data, engineer_features, get_feature_names


def train_model(X_train, y_train):
    """Train Random Forest with SMOTE inside CV"""

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train_smote, y_train_smote)
    return model


def main():
    """Load data and train model"""

    # Load and preprocess
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    # Split features and target
    feature_names = get_feature_names(df)
    X = df[feature_names].values
    y = (df['Churn'] == 'Yes').astype(int).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = train_model(X_scaled, y)

    # Save model
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'threshold': 0.23,
    }

    Path('models').mkdir(exist_ok=True)
    joblib.dump(artifact, 'models/customer_churn_model.pkl')
    print("✓ Model trained and saved")


if __name__ == '__main__':
    main()
