"""Train the selected production churn model."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from pathlib import Path


def train_model(X_train, y_train):
    """Train Logistic Regression with scaling and SMOTE."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def main():
    """Load data and train model"""
    df = pd.read_parquet("data/processed/train.parquet")

    # Split features and target
    feature_names = [col for col in df.columns if col != "Churn"]
    X = df[feature_names]
    y = df["Churn"]

    # Train model
    model = train_model(X, y)

    # Save model
    artifact = {
        "model": model,
        "scaler": None,
        "feature_names": feature_names,
        "threshold": 0.28,
        "model_name": "logistic_regression_churn_classifier",
        "training_date": "2026-05-02",
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(artifact, "models/customer_churn_model.pkl")
    print("Model trained and saved")


if __name__ == "__main__":
    main()
