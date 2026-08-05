from pathlib import Path
import pandas as pd
from common.utils import ensure_dir
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    if format_type == "F12":
        le = LabelEncoder()
        df["defect_label"] = le.fit_transform(df["defect_type"].fillna("Unknown"))
        pivot = df.pivot_table(index="die_y", columns="die_x", values="bin_code", fill_value=0)
    else:  # F8
        max_r, max_c = df["row"].max()+1, df["col"].max()+1
        pivot = np.zeros((max_r, max_c))
        pivot[df["row"], df["col"]] = df["bin_code"]
        # 補零還原圓形（依 wafer 半徑 mask）
    return pivot.values if isinstance(pivot, pd.DataFrame) else pivot