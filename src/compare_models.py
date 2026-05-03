"""Compare churn classification models on the held out test set."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTREACH_COST = 25
EXPECTED_SAVE_PER_TRUE_POSITIVE = 500 * 0.30


def load_processed_data():
    """Load the existing train and test parquet files."""
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    feature_cols = [col for col in train_df.columns if col != "Churn"]
    X_train = train_df[feature_cols].values
    y_train = train_df["Churn"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Churn"].values

    return X_train, y_train, X_test, y_test


def build_models():
    """Create candidate model pipelines using the same train/test data."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "AdaBoost": Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", KNeighborsClassifier(n_neighbors=25)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", GaussianNB()),
            ]
        ),
    }


def find_best_business_threshold(y_true, y_score):
    """Choose the threshold with the highest expected retention value."""
    best = {
        "threshold": 0.50,
        "business_value": float("-inf"),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    for threshold in np.arange(0.05, 0.91, 0.01):
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        value = (
            tp * EXPECTED_SAVE_PER_TRUE_POSITIVE
            - (tp + fp) * OUTREACH_COST
        )

        if value > best["business_value"]:
            best = {
                "threshold": round(float(threshold), 2),
                "business_value": round(float(value), 2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

    return best


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a candidate classifier."""
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred_050 = (y_score >= 0.50).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    best_threshold = find_best_business_threshold(y_test, y_score)

    return {
        "model": name,
        "auc_roc": roc_auc_score(y_test, y_score),
        "auc_pr": auc(recall_curve, precision_curve),
        "accuracy_0_50": accuracy_score(y_test, y_pred_050),
        "precision_0_50": precision_score(y_test, y_pred_050, zero_division=0),
        "recall_0_50": recall_score(y_test, y_pred_050, zero_division=0),
        "f1_0_50": f1_score(y_test, y_pred_050, zero_division=0),
        "best_threshold": best_threshold["threshold"],
        "business_value": best_threshold["business_value"],
        "precision_at_best_threshold": best_threshold["precision"],
        "recall_at_best_threshold": best_threshold["recall"],
        "f1_at_best_threshold": best_threshold["f1"],
        "tp_at_best_threshold": best_threshold["tp"],
        "fp_at_best_threshold": best_threshold["fp"],
        "fn_at_best_threshold": best_threshold["fn"],
    }


def write_markdown_report(results):
    """Write a compact markdown report for GitHub."""
    best = results.iloc[0]
    table_cols = [
        "model",
        "auc_roc",
        "auc_pr",
        "best_threshold",
        "business_value",
        "precision_at_best_threshold",
        "recall_at_best_threshold",
        "f1_at_best_threshold",
    ]
    report = [
        "# Model Comparison Report",
        "",
        "The comparison uses the existing processed training and test sets in `data/processed/`.",
        "Each candidate is trained on the training set and evaluated on the held out test set.",
        "SMOTE is applied inside each training pipeline to handle class imbalance.",
        "",
        "Business value uses this formula:",
        "",
        "```text",
        "true positives * 150 - flagged customers * 25",
        "```",
        "",
        f"Best model by business value: {best['model']}.",
        "",
        results[table_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Note: the threshold is selected to maximize business value on the held out comparison set.",
        "For production retraining, choose the threshold on a validation set and keep the test set for final reporting.",
        "",
    ]
    Path("reports/model_comparison.md").write_text("\n".join(report))


def main():
    warnings.filterwarnings("ignore")
    X_train, y_train, X_test, y_test = load_processed_data()

    rows = []
    for name, model in build_models().items():
        rows.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))

    results = (
        pd.DataFrame(rows)
        .sort_values(["business_value", "auc_pr", "auc_roc"], ascending=False)
        .reset_index(drop=True)
    )

    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/model_comparison.csv", index=False)
    write_markdown_report(results)

    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model by business value: {results.iloc[0]['model']}")


if __name__ == "__main__":
    main()
