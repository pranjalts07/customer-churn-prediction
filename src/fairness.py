"""Fairness and bias audit."""
import joblib
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


def main():
    """Run a compact fairness audit for the saved production model."""
    artifact = joblib.load("models/customer_churn_model.pkl")
    test_df = pd.read_parquet("data/processed/test.parquet")
    feature_names = artifact["feature_names"]
    threshold = artifact.get("threshold", 0.28)

    probabilities = artifact["model"].predict_proba(test_df[feature_names])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    y_true = test_df["Churn"].to_numpy()

    parity = demographic_parity_audit(test_df, predictions)
    odds = equalized_odds_audit(test_df, y_true, predictions, sensitive_attr="gender")

    print("Demographic parity audit")
    for attr, result in parity.items():
        rates = ", ".join(f"{group}: {rate:.3f}" for group, rate in result["group_rates"].items())
        print(f"{attr}: {rates}; disparity {result['disparity']:.3f}; acceptable {result['acceptable']}")

    print("\nEqualized odds audit by gender")
    for group, result in odds.items():
        print(f"{group}: TPR {result['tpr']:.3f}; FPR {result['fpr']:.3f}")


if __name__ == "__main__":
    main()
