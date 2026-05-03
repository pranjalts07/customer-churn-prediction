"""Model calibration analysis"""
import numpy as np
from sklearn.metrics import brier_score_loss


def calculate_brier_score(y_true, y_pred_proba):
    """Calculate Brier score (expected calibration error for binary)"""
    return brier_score_loss(y_true, y_pred_proba)


def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """Compute ECE: average difference between predicted and actual"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    total_samples = len(y_true)

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > lower) & (y_pred_proba <= upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece
