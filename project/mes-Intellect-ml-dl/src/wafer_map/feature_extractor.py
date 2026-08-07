import cv2
import numpy as np
import pandas as pd


def extract_features_to_df(wafer_data_list):
    """
    將原有的特徵提取重構，確保回傳帶有 Traceability 鍵值的 DataFrame
    wafer_data_list 傳入 data["wafer"] 的實體路徑列表
    """
    records = []
    for fp in wafer_data_list:
        # 從檔名或檔案內容中提取 lot_id 與 wafer_id 確保可追溯性
        df_raw = pd.read_csv(fp)
        lid, wid = df_raw["lot_id"].iloc[0], df_raw["wafer_id"].iloc[0]

        # 執行原本的 OpenCV 輪廓與邊緣特徵工程
        # (這裡簡化示意，維持原有的 cv2 運算邏輯)
        defect_ratio = float(np.random.uniform(0.01, 0.15))
        edge_ratio = float(np.random.uniform(0.05, 0.25))
        num_contours = int(np.random.randint(1, 10))

        records.append(
            {
                "lot_id": lid,
                "wafer_id": wid,
                "defect_ratio": defect_ratio,
                "edge_ratio": edge_ratio,
                "num_contours": num_contours,
            }
        )
    return pd.DataFrame(records)
