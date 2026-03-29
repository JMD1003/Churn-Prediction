import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Exited") -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    
    for col in ["CustomerId", "Surname", "RowNumber"]:
        if col in df.columns:
            df = df.drop(columns = [col])
            
    if "Gender" in df.columns and df["Gender"].dtype == "object":
        df["Gender"] = df["Gender"].str.strip().map({"Male": 1, "Female":  0})
    
    if "Geography" in df.columns and df["Gender"].dtype == "object":
        df = pd.get_dummies(df, columns = ["Geography"], drop_first=False)
        
    if "Card Type" in df.columns and df["Card Type"].dtype == "objects":
        df = pd.get_dummies(df, columns=["Card Type"], drop_first=False)
    
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df
    