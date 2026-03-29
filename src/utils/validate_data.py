import great_expectations as ge 
from typing import Tuple, List

def validate_data_churn_data(df) -> Tuple[bool, list[str]]:
    
    print("Starting data visulaidation with Great Expectations...")
    
    ge_df = ge.dataset.PandasDataset(df)
    
    print("Validating schema and required columns...")
    
    ge_df.expect_column_to_exist("CustomerId")
    ge_df.expect_column_values_to_not_be_null("CustomerId")
    
    ge_df.expect_column_to_exist("Gender")
    ge_df.expect_column_to_exist("Age")
    ge_df.expect_column_to_exist("Geography")

    ge_df.expect_column_to_exist("CreditScore")
    ge_df.expect_column_to_exist("Balance")
    ge_df.expect_column_to_exist("EstimatedSalary")
    ge_df.expect_column_to_exist("Tenure")
    ge_df.expect_column_to_exist("NumOfProducts")

    ge_df.expect_column_to_exist("HasCrCard")
    ge_df.expect_column_to_exist("IsActiveMember")
    ge_df.expect_column_to_exist("Complain")
    ge_df.expect_column_to_exist("Satisfaction Score")
    ge_df.expect_column_to_exist("Point Earned")
    ge_df.expect_column_to_exist("Card Type")

    ge_df.expect_column_to_exist("Exited")

    print("Validating business logic contraints...")

    ge_df.expect_column_values_to_be_in_set("Gender", ["Male", "Female"])

    ge_df.expect_column_values_to_be_in_set("Geography", ["France", "Germany", "Spain"])

    ge_df.expect_column_values_to_be_in_set("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])

    ge_df.expect_column_values_to_be_in_set("HasCrCard", [0, 1])
    ge_df.expect_column_values_to_be_in_set("IsActiveMember", [0, 1])
    ge_df.expect_column_values_to_be_in_set("Complain", [0, 1])
    ge_df.expect_column_values_to_be_in_set("Exited", [0, 1])

    ge_df.expect_column_values_to_be_in_set("NumOfProducts", [1, 2, 3, 4])

    print("Validating numeric ranges and business constraints...")

    ge_df.expect_column_values_to_be_between("CreditScore", min_value=300, max_value=850)

    ge_df.expect_column_values_to_be_between("Age", min_value=18, max_value=100)

    ge_df.expect_column_values_to_be_between("Tenure", min_value=0, max_value=10)

    ge_df.expect_column_values_to_be_between("Balance", min_value=0)

    ge_df.expect_column_values_to_be_between("EstimatedSalary", min_value=0)

    ge_df.expect_column_values_to_be_between("Satisfaction Score", min_value=1, max_value=5)

    ge_df.expect_column_values_to_be_between("Point Earned", min_value=0)

    print("   🔍 Checking for nulls in critical columns...")

    for col in ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
                "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]:
        ge_df.expect_column_values_to_not_be_null(col)

    print("Running complete validation suite...")
    results = ge_df.validate()

    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations