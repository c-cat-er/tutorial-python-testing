from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common.utils import ensure_dir


def load_all_sources():
    """統一載入 MES、NPI、Wafer Map 資料"""
    base = Path("data/raw")
    return {
        "mes": list(base.glob("mes_logs/*.log")),
        "npi": list(base.glob("npi/*.csv")) + list(base.glob("npi/*.txt")),
        "wafer": list(base.glob("wafer_maps/*.csv")),
    }


def wafer_to_matrix(csv_path, format_type="F12"):
    df = pd.read_csv(csv_path)

    if "die_y" in df.columns and "die_x" in df.columns:
        df = df.dropna(subset=["die_y", "die_x", "bin_code"])
        df["die_y"] = df["die_y"].astype(int)
        df["die_x"] = df["die_x"].astype(int)
        pivot = df.pivot_table(
            index="die_y", columns="die_x", values="bin_code", fill_value=0
        )
        matrix = pivot.values.astype(np.uint8)
    elif "row" in df.columns and "col" in df.columns:
        df = df.dropna(subset=["row", "col", "bin_code"])
        max_r, max_c = int(df["row"].max() + 1), int(df["col"].max() + 1)
        pivot = np.zeros((max_r, max_c), dtype=np.uint8)
        pivot[df["row"].astype(int), df["col"].astype(int)] = df["bin_code"].astype(
            np.uint8
        )
        matrix = pivot
    else:
        raise KeyError(f"無法辨識的 WaferMap 格式: {df.columns.tolist()}")

    # 移除了疊加 3 通道的邏輯，嚴格維持 2D 矩陣狀態回傳
    return matrix
