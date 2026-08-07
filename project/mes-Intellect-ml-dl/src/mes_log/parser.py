import re
from datetime import datetime
from pathlib import Path

from src.common.data_models import LogEntry

# 重構 Regex，使其精準捕捉實體 Log 中的所有必要欄位
LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|"
    r"(?P<lot_id>[^|]+)\|"
    r"(?P<wafer_id>[^|]+)\|"
    r"(?P<equipment_id>[^|]+)\|"
    r"(?P<process_step>[^|]+)\|"
    r"(?P<event_type>[^|]+)\|"
    r"(?P<message>[^|]+)\|"
    r"(?P<operator>[^|]+)\|"
    r"(?P<status>[^|\n]+)"
)


def parse_mes_logs(file_paths):
    results = []
    severity_map = {"INFO": 0, "WARN": 1, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # 跳過 CSV/Log 的標頭列 (Header)
                if "timestamp|lot_id" in line:
                    continue
                if not line.strip():
                    continue

                match = LOG_PATTERN.search(line.strip())
                if match:
                    entry = match.groupdict()
                    entry["raw"] = line.strip()
                    entry["file"] = Path(file_path).name

                    # 雖然日誌中沒有獨立的 level 欄位，但依據產線異常邏輯：
                    # 如果訊息中出現 Overlay shift 或重大異常，則將 level 定義為 CRITICAL，否則為 INFO
                    is_anomaly = (
                        "shift" in entry["message"].lower()
                        or "warning" in entry["event_type"].lower()
                    )
                    entry["level"] = "CRITICAL" if is_anomaly else "INFO"
                    entry["severity"] = severity_map.get(entry["level"], 0)

                    # 將 process_step 映射回 Pydantic 模型寫死的 module 屬性，確保相容
                    entry["module"] = entry["process_step"]

                    # 欄位全數對齊歸位，Pydantic 高興放行！
                    results.append(LogEntry(**entry))
    return results
