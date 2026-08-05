import numpy as np
import pandas as pd


def ensure_dir(path: str):
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)


def standardize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    for std_col, possibles in mapping.items():
        for col in df.columns:
            if any(p.lower() in col.lower() for p in possibles):
                df.rename(columns={col: std_col}, inplace=True)
    return df


def prepare_feature_matrix(mes_df, npi_df, wafer_features_df):
    """
    透過樞紐轉置（Pivot），將 NPI 縱向的多個量測橫列，
    轉換為橫向並排的機器學習特徵，徹底解決二維 DataFrame 膨脹問題。
    """
    keys = ["lot_id", "wafer_id"]

    # 1. 複製資料，避免改動到全域變數
    mes_copy = mes_df.copy()
    npi_copy = npi_df.copy()
    wafer_copy = wafer_features_df.copy()

    # 2. 欄位標頭強制轉小寫
    mes_copy.columns = [str(c).lower() for c in mes_copy.columns]
    npi_copy.columns = [str(c).lower() for c in npi_copy.columns]
    wafer_copy.columns = [str(c).lower() for c in wafer_copy.columns]

    # 防禦機制，移除因為對齊與大小寫轉換產生的重複欄位
    npi_copy = npi_copy.loc[:, ~npi_copy.columns.duplicated()]

    # 3. 確保數值型態正確
    npi_copy["value"] = pd.to_numeric(npi_copy["value"], errors="coerce")
    npi_copy["yield_pct"] = pd.to_numeric(npi_copy["yield_pct"], errors="coerce")

    # 4. 🔴【核心修正】利用樞紐轉置將 ctq_parameter 展開成橫向欄位
    # 這會將複數個橫列自動壓平成唯一的晶圓 Row，並產出 cd_linewidth 與 thickness 獨立欄位
    npi_pivot = npi_copy.pivot_table(
        index=keys, columns="ctq_parameter", values="value", aggfunc="mean"
    ).reset_index()
    npi_pivot.columns = [str(c).lower() for c in npi_pivot.columns]  # 轉小寫對齊

    # 5. 單獨對良率取平均（每片晶圓唯一一列）
    npi_yield = npi_copy.groupby(keys, as_index=False)["yield_pct"].mean()
    npi_agg = pd.merge(npi_pivot, npi_yield, on=keys, how="inner")

    # 6. 聚合 MES（抓最大嚴重度）與 WaferMap 特徵（抓首列）
    mes_agg = mes_copy.groupby(keys, as_index=False)[["severity"]].max()
    wafer_agg = wafer_copy.groupby(keys, as_index=False)[
        ["defect_ratio", "edge_ratio", "num_contours"]
    ].first()

    # 7. 三方標準扁平表格進行 Inner Join 合併
    merged = pd.merge(npi_agg, mes_agg, on=keys, how="inner")
    merged = pd.merge(merged, wafer_agg, on=keys, how="inner")

    # 8. 重新定義全新、符合產線物理意義的特徵欄位列表
    # 這裡將舊的 "value" 改為實體的 "cd_linewidth" 與 "thickness"
    feature_cols = [
        "severity",
        "cd_linewidth",
        "thickness",
        "defect_ratio",
        "edge_ratio",
        "num_contours",
    ]

    # 防禦檢查：若某批次剛好缺某參數，自動補 0
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    # 9. 封裝成 AI 模型專用的 2D 矩陣
    X = merged.set_index(keys)[feature_cols].astype(np.float32)
    y = merged.set_index(keys)["yield_pct"].values.astype(np.float32)

    return X, y
