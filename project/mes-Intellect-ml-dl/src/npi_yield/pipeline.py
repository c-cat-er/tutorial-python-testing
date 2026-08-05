# src/npi_yield/pipeline.py
import pandas as pd

from src.common.utils import standardize_columns


def run_npi_pipeline(file_paths, spec_limits, column_mapping):
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, engine="pyarrow", on_bad_lines="skip")
        df = standardize_columns(df, column_mapping)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # 使用外部傳入的 spec_limits 進行 3-Sigma / Outlier 標記
    for col, limits in spec_limits.items():
        if col in df_all.columns:
            df_all[f"{col}_outlier"] = (df_all[col] > limits["USL"]) | (
                df_all[col] < limits["LSL"]
            )
    return df_all
