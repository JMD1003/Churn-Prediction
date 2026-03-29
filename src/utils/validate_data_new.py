import pandas as pd
from typing import Tuple, List


def validate_data_churn_data_new(df) -> Tuple[bool, List[str]]:
    print("Starting data validation...")

    failed = []

    # Schema checks
    required_cols = [
        "CustomerId", "Gender", "Age", "Geography", "CreditScore",
        "Balance", "EstimatedSalary", "Tenure", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "Complain",
        "Satisfaction Score", "Point Earned", "Card Type", "Exited"
    ]
    for col in required_cols:
        if col not in df.columns:
            failed.append(f"missing_column: {col}")

    # Null checks
    for col in ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary", "Exited"]:
        if col in df.columns and df[col].isna().sum() > 0:
            failed.append(f"nulls_in: {col}")

    # Value set checks
    if "Gender" in df.columns:
        invalid = ~df["Gender"].isin(["Male", "Female"])
        if invalid.any():
            failed.append("invalid_values: Gender")

    if "Geography" in df.columns:
        invalid = ~df["Geography"].isin(["France", "Germany", "Spain"])
        if invalid.any():
            failed.append("invalid_values: Geography")

    if "Card Type" in df.columns:
        invalid = ~df["Card Type"].isin(["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
        if invalid.any():
            failed.append("invalid_values: Card Type")

    # Range checks
    if "CreditScore" in df.columns:
        if not df["CreditScore"].between(300, 850).all():
            failed.append("out_of_range: CreditScore")

    if "Age" in df.columns:
        if not df["Age"].between(18, 100).all():
            failed.append("out_of_range: Age")

    if "Tenure" in df.columns:
        if not df["Tenure"].between(0, 10).all():
            failed.append("out_of_range: Tenure")

    is_valid = len(failed) == 0

    if is_valid:
        print(f"Data validation PASSED")
    else:
        print(f"Data validation FAILED: {failed}")

    return is_valid, failed