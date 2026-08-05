import os
from pathlib import Path
import pandas as pd

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def standardize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    for std_col, possibles in mapping.items():
        for col in df.columns:
            if any(p.lower() in col.lower() for p in possibles):
                df.rename(columns={col: std_col}, inplace=True)
    return df