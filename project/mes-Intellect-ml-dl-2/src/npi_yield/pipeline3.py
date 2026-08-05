from pathlib import Path

import pandas as pd
import yaml

from src.common.utils import standardize_columns

# 1. 取得全局專案根目錄，並定位最外層的 config/settings.yaml
project_root = Path(__file__).resolve().parent.parent.parent
config_path = project_root / "config" / "settings.yaml"

# 2. 現場讀取 YAML 設定檔
with open(config_path, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

# 3. 從 settings.yaml 內的 npi 區塊動態撈出資料 (完全不寫死)
npi_config = settings.get("npi", {})
COLUMN_MAPPING = npi_config.get("column_mapping", {})
SPEC_LIMITS = npi_config.get(
    "spec_limits",
    {
        "CD_LINEWIDTH": {"USL": 0.095, "LSL": 0.075},
        "THICKNESS": {"USL": 255.0, "LSL": 235.0},
        "OVERLAY_X": {"USL": 0.030, "LSL": -0.030},
    },
)


def run_npi_pipeline(file_paths):
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, engine="pyarrow", on_bad_lines="skip")
        df = standardize_columns(df, COLUMN_MAPPING)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # 4. 根據 settings.yaml 讀出來的界限，進行 3-Sigma/產線 outlier 標記
    for col, limits in SPEC_LIMITS.items():
        if col in df_all.columns:
            df_all[f"{col}_outlier"] = (df_all[col] > limits["USL"]) | (
                df_all[col] < limits["LSL"]
            )
    return df_all
