import pandas as pd

def clean_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the case where 'timestamp' exists both as index and column.
    Ensures timestamp is only in the index, and unnamed.
    """
    if "timestamp" in df.columns and df.index.name == "timestamp":
        del df["timestamp"]
        df.index.name = None
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index.name = None
    elif df.index.name == "timestamp":
        df.index.name = None
    return df
