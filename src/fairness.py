"""Fairness and bias audit"""
import pandas as pd
import numpy as np


def demographic_parity_audit(df, predictions, sensitive_attrs=['gender', 'SeniorCitizen']):
    """
    Check for demographic parity violations
    Demographic parity: P(prediction=1 | group=A) ≈ P(prediction=1 | group=B)
    """
    results = {}

    for attr in sensitive_attrs:
        if attr not in df.columns:
            continue

        groups = df[attr].unique()
        rates = {}

        for group in groups:
            group_mask = df[attr] == group
            positive_rate = predictions[group_mask].mean()
            rates[group] = positive_rate

        # Calculate disparity ratio
        if len(rates) == 2:
            groups_list = list(rates.keys())
            disparity = abs(rates[groups_list[0]] - rates[groups_list[1]]) / (rates[groups_list[1]] + 1e-6)
            results[attr] = {
                'group_rates': rates,
                'disparity': disparity,
                'acceptable': disparity < 0.1
            }

    return results


def equalized_odds_audit(df, y_true, predictions, sensitive_attr='gender'):
    """
    Check equalized odds: FPR and TPR should be equal across groups
    """
    groups = df[sensitive_attr].unique()
    results = {}

    for group in groups:
        group_mask = df[sensitive_attr] == group

        tp = ((predictions[group_mask] == 1) & (y_true[group_mask] == 1)).sum()
        fp = ((predictions[group_mask] == 1) & (y_true[group_mask] == 0)).sum()
        fn = ((predictions[group_mask] == 0) & (y_true[group_mask] == 1)).sum()
        tn = ((predictions[group_mask] == 0) & (y_true[group_mask] == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results[group] = {'tpr': tpr, 'fpr': fpr}

    return results
