import pandas as pd

from src.common.data_models import LogEntry


def clean_mes_logs(log_entries: list[LogEntry]):
    df = pd.DataFrame([e.dict() for e in log_entries])
    severity_map = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}
    df["severity"] = df["level"].map(severity_map).fillna(0).astype(int)

    # 過濾重複與無效資料
    df = df.drop_duplicates(subset=["timestamp", "message"])
    return df
