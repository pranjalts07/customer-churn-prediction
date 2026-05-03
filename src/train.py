"""Train, evaluate, and save the selected production churn model."""
from datetime import date
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


THRESHOLD = 0.28
OUTREACH_COST = 25
EXPECTED_SAVE_PER_TRUE_POSITIVE = 500 * 0.30
MODEL_PATH = Path("models/customer_churn_model.pkl")
MODEL_CARD_PATH = Path("models/model_card.json")
PRIORITY_LIST_PATH = Path("reports/priority_contact_list.csv")


def train_model(X_train, y_train):
    """Train Logistic Regression with scaling and SMOTE inside the pipeline."""
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


def score_risk_tier(probability):
    """Convert churn probability to a business risk tier."""
    if probability >= 0.70:
        return "CRITICAL"
    if probability >= 0.40:
        return "AT-RISK"
    if probability >= THRESHOLD:
        return "WATCH"
    return "SAFE"


def recommended_action(tier):
    """Return the recommended retention action for a risk tier."""
    if tier == "CRITICAL":
        return "Immediate retention call with contract or loyalty offer"
    if tier == "AT-RISK":
        return "Priority outreach with targeted discount or service bundle"
    if tier == "WATCH":
        return "Monitor closely and send proactive retention email"
    return "No action needed"


def evaluate_on_test(model, test_df, feature_names):
    """Calculate model, threshold, and business metrics on the holdout set."""
    X_test = test_df[feature_names]
    y_test = test_df["Churn"]
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    tp = int(((predictions == 1) & (y_test == 1)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())
    flagged = int(predictions.sum())
    business_value = tp * EXPECTED_SAVE_PER_TRUE_POSITIVE - flagged * OUTREACH_COST

    monthly = test_df["MonthlyCharges_raw"]
    revenue_at_risk = probabilities * monthly.to_numpy() * 24
    priority_mask = probabilities >= THRESHOLD
    tiers = pd.Series(probabilities).map(score_risk_tier)

    metrics = {
        "auc_roc": roc_auc_score(y_test, probabilities),
        "auc_pr": auc(recall_curve, precision_curve),
        "accuracy_at_threshold": accuracy_score(y_test, predictions),
        "precision_at_threshold": precision_score(y_test, predictions, zero_division=0),
        "recall_at_threshold": recall_score(y_test, predictions, zero_division=0),
        "f1_at_threshold": f1_score(y_test, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_test, probabilities),
        "threshold": THRESHOLD,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flagged": flagged,
        "business_value": business_value,
        "revenue_at_risk_usd": float(revenue_at_risk[priority_mask].sum()),
        "potential_saves_usd": float(revenue_at_risk[priority_mask].sum() * 0.30),
        "risk_tiers": tiers[priority_mask].value_counts().to_dict(),
        "safe_customers": int((tiers == "SAFE").sum()),
    }

    priority_df = pd.DataFrame(
        {
            "customer_id": range(1, len(test_df) + 1),
            "risk_tier": tiers,
            "churn_probability": probabilities,
            "revenue_at_risk": revenue_at_risk,
            "monthly_charges": monthly.to_numpy(),
        }
    )
    priority_df = (
        priority_df[priority_df["churn_probability"] >= THRESHOLD]
        .sort_values(["churn_probability", "revenue_at_risk"], ascending=False)
        .reset_index(drop=True)
    )
    priority_df.insert(0, "rank", range(1, len(priority_df) + 1))
    priority_df["recommended_action"] = priority_df["risk_tier"].map(recommended_action)

    return metrics, priority_df


def write_model_card(train_df, test_df, metrics):
    """Write GitHub-readable metadata for the production model."""
    model_card = {
        "model": {
            "name": "Customer Churn Prediction",
            "version": "1.0.0",
            "type": "LogisticRegression",
            "trained_at": date.today().isoformat(),
        },
        "training": {
            "dataset": "IBM Telco Customer Churn",
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "churn_rate_train": round(float(train_df["Churn"].mean()), 3),
            "churn_rate_test": round(float(test_df["Churn"].mean()), 3),
            "features_count": int(len([col for col in train_df.columns if col != "Churn"])),
            "feature_engineering": [
                "is_month_to_month",
                "is_honeymoon_period",
                "monthly_charge_per_tenure",
                "service_bundle_depth",
                "unprotected_fiber",
            ],
        },
        "performance": {
            "auc_roc": round(float(metrics["auc_roc"]), 4),
            "auc_pr": round(float(metrics["auc_pr"]), 4),
            "precision_at_threshold_0_28": round(float(metrics["precision_at_threshold"]), 4),
            "recall_at_threshold_0_28": round(float(metrics["recall_at_threshold"]), 4),
            "f1_at_threshold_0_28": round(float(metrics["f1_at_threshold"]), 4),
            "brier_score": round(float(metrics["brier_score"]), 4),
        },
        "business": {
            "threshold": THRESHOLD,
            "cost_optimization_ratio": "20:1 missed_churn vs outreach_cost",
            "customers_identified": int(metrics["flagged"]),
            "revenue_at_risk_usd": round(float(metrics["revenue_at_risk_usd"])),
            "potential_saves_usd": round(float(metrics["potential_saves_usd"])),
            "assumed_retention_rate": 0.30,
        },
        "limitations": [
            "Threshold selected for business value on held-out comparison set",
            "Tenure-based holdout has a lower churn rate than the training set",
            "CLV model uses a simplified 24-month value window",
        ],
        "next_steps": [
            "Shadow deployment",
            "A/B testing",
            "Periodic retraining and drift monitoring",
        ],
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2) + "\n")


def main():
    """Load processed data, train the production model, and write artifacts."""
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    feature_names = [col for col in train_df.columns if col != "Churn"]

    model = train_model(train_df[feature_names], train_df["Churn"])
    metrics, priority_df = evaluate_on_test(model, test_df, feature_names)

    metadata = {
        "project_name": "Customer Churn Prediction",
        "selected_by": "highest held-out business value and AUC PR in model comparison",
        "test_metrics": metrics,
        "candidate_models_tested": [
            "Logistic Regression",
            "Gradient Boosting",
            "Random Forest",
            "AdaBoost",
            "Decision Tree",
            "KNN",
            "Extra Trees",
            "Gaussian Naive Bayes",
        ],
    }
    artifact = {
        "model": model,
        "threshold": THRESHOLD,
        "feature_names": feature_names,
        "model_name": "logistic_regression_churn_classifier",
        "training_date": date.today().isoformat(),
        "metadata": metadata,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    PRIORITY_LIST_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    priority_df.to_csv(PRIORITY_LIST_PATH, index=False)
    write_model_card(train_df, test_df, metrics)

    print("Model trained and saved")
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC PR: {metrics['auc_pr']:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Customers above threshold: {metrics['flagged']:,}")
    print(f"Revenue at risk: ${metrics['revenue_at_risk_usd']:,.0f}")


if __name__ == "__main__":
    main()
