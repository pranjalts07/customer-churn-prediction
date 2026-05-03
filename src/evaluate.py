"""Model evaluation and threshold optimization"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
