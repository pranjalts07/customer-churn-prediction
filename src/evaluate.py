"""Model evaluation and threshold optimization."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib


def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model performance"""

    if scaler:
        X_test = scaler.transform(X_test)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'y_pred_proba': y_pred_proba
    }


def optimize_threshold(y_test, y_pred_proba, cost_fp=25, cost_fn=500*0.30):
    """
    Find optimal threshold using cost-weighted formula
    EV(threshold) = (TP × 0.30 × $500) - ((TP + FP) × $25)
    """

    best_threshold = 0.5
    best_ev = float('-inf')

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()

        ev = (tp * cost_fn) - ((tp + fp) * cost_fp)

        if ev > best_ev:
            best_ev = ev
            best_threshold = threshold

    return best_threshold


def get_metrics_at_threshold(y_test, y_pred_proba, threshold):
    """Get precision, recall at specific threshold"""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp}


def main():
    """Evaluate the saved production model on the processed test set."""
    artifact = joblib.load("models/customer_churn_model.pkl")
    test_df = pd.read_parquet("data/processed/test.parquet")
    feature_names = artifact["feature_names"]
    threshold = artifact.get("threshold", 0.28)

    X_test = test_df[feature_names]
    y_test = test_df["Churn"]
    probabilities = artifact["model"].predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = evaluate_model(artifact["model"], X_test, y_test)
    threshold_metrics = get_metrics_at_threshold(y_test, probabilities, threshold)

    print(f"Model: {artifact.get('model_name', 'customer_churn_model')}")
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC PR: {metrics['auc_pr']:.4f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"True positives: {threshold_metrics['tp']}")
    print(f"False positives: {threshold_metrics['fp']}")
    print(f"Flagged customers: {int(predictions.sum())}")


if __name__ == "__main__":
    main()
