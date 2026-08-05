from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel


class TraceabilityID(BaseModel):
    lot_id: str
    wafer_id: Optional[str] = None
    die_id: Optional[str] = None


class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    module: Optional[str]
    lot_id: str
    wafer_id: Optional[str] = None
    equipment_id: Optional[str] = None
    message: str
    raw: str
    file: str
    severity: int


class WaferBinMap(BaseModel):
    lot_id: str
    wafer_id: str
    bin_map: List[List[int]]
    defect_type: Optional[str] = None


# def predict_yield(mes_df, npi_df, wafer_features):
#     # 1. 修正對齊鍵值，納入說明書要求的 Traceability ID 規範
#     keys = ["lot_id", "wafer_id", "die_id"]  # 若到 Die 層級，需確保資料顆粒度對齊

#     # 確保進行 Inner Join 時，數據能精準對齊到單顆 Die 的層級
#     X_df = mes_df.merge(npi_df, on=keys, how="inner")
#     npi_df = npi_df.merge(mes_df[keys], on=keys, how="inner")

#     X = pd.concat(
#         [
#             mes_df.set_index(keys)[["severity"]],
#             npi_df.set_index(keys)[["Value"]],
#             pd.DataFrame(wafer_features, index=mes_df.set_index(keys).index),
#         ],
#         axis=1,
#     )

#     # 模擬良率標籤 (實際場景應從 NPI Data 的 yield_pct 取得)
#     y = np.random.uniform(80, 100, size=len(X))

#     # 2. 完整補齊模型訓練與預測
#     model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
#     model.fit(X, y)
#     preds = model.predict(X)

#     # 3. 確保函數有完整 Output Return
#     return {
#         "predicted_yield": float(preds.mean()),
#         "feature_importances": model.feature_importances_.tolist(),
#     }
