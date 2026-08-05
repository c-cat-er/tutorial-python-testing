import pandas as pd
from common.utils import standardize_columns
from config import COLUMN_MAPPING, SPEC_LIMITS

def run_npi_pipeline(file_paths):
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, encoding='utf-8-sig', on_bad_lines='skip')
        df = standardize_columns(df, COLUMN_MAPPING)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    # 加入 3-Sigma / USL/LSL 標記
    for col, limits in SPEC_LIMITS.items():
        if col in df_all.columns:
            df_all[f'{col}_outlier'] = (df_all[col] > limits['USL']) | (df_all[col] < limits['LSL'])
    return df_all