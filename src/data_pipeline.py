"""Build processed train and test sets for the churn project."""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DIR = Path("data/processed")
TRAIN_SIZE = 5282

BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

DUMMY_COLUMNS = ["Contract", "InternetService", "PaymentMethod"]
SCALED_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "monthly_charge_per_tenure"]


def load_raw_data(data_path=RAW_DATA_PATH):
    """Load the IBM Telco Churn dataset."""
    return pd.read_csv(data_path)


def clean_raw_data(df):
    """Clean labels, charges, and binary customer attributes."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["MonthlyCharges_raw"] = df["MonthlyCharges"]

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).astype(int)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    for column in BINARY_COLUMNS:
        df[column] = df[column].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    return df


def engineer_features(df):
    """Create business features used by the production model."""
    df = df.copy()
    df["is_honeymoon_period"] = (df["tenure"] < 6).astype(int)
    df["unprotected_fiber"] = (
        (df["InternetService"] == "Fiber optic")
        & (df["OnlineSecurity"] == 0)
        & (df["TechSupport"] == 0)
    ).astype(int)
    df["monthly_charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["service_bundle_depth"] = df[service_columns].sum(axis=1)
    df["is_auto_pay"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def encode_features(df):
    """One-hot encode multi-class fields and order columns for training."""
    df = df.copy()
    encoded = pd.get_dummies(df[DUMMY_COLUMNS], columns=DUMMY_COLUMNS)
    df = pd.concat([df.drop(columns=["customerID"] + DUMMY_COLUMNS), encoded], axis=1)

    ordered_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "Contract_Month-to-month",
        "Contract_One year",
        "Contract_Two year",
        "InternetService_DSL",
        "InternetService_Fiber optic",
        "InternetService_No",
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
        "is_honeymoon_period",
        "unprotected_fiber",
        "monthly_charge_per_tenure",
        "service_bundle_depth",
        "is_auto_pay",
        "is_month_to_month",
        "MonthlyCharges_raw",
    ]
    return df[ordered_columns]


def split_and_scale(df):
    """Create the current tenure-based holdout and scale numeric columns."""
    df = df.sort_values("tenure", kind="quicksort").reset_index(drop=True)
    train_df = df.iloc[:TRAIN_SIZE].copy()
    test_df = df.iloc[TRAIN_SIZE:].copy()

    scaler = StandardScaler()
    train_df[SCALED_COLUMNS] = scaler.fit_transform(train_df[SCALED_COLUMNS])
    test_df[SCALED_COLUMNS] = scaler.transform(test_df[SCALED_COLUMNS])

    return train_df, test_df


def build_processed_data(raw_df):
    """Run the full preprocessing flow and return train and test frames."""
    df = clean_raw_data(raw_df)
    df = engineer_features(df)
    df = encode_features(df)
    return split_and_scale(df)


def main():
    """Write processed parquet files used by training and comparison scripts."""
    raw_df = load_raw_data()
    train_df, test_df = build_processed_data(raw_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print(f"Processed training rows: {len(train_df):,}")
    print(f"Processed test rows: {len(test_df):,}")
    print(f"Training churn rate: {train_df['Churn'].mean():.3f}")
    print(f"Test churn rate: {test_df['Churn'].mean():.3f}")


if __name__ == "__main__":
    main()
