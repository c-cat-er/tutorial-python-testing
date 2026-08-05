import logging
from pathlib import Path
from common.database import SessionLocal
from acquisition.loader import load_all_sources
from mes_log.parser import parse_mes_logs
from mes_log.cleaner import clean_mes_logs
from npi_yield.pipeline import run_npi_pipeline
from wafer_map.feature_extractor import extract_features
from wafer_map.classifier import classify_defects
from yield_predictor.predictor import predict_yield
from analysis.reporting import generate_report
from acquisition.loader import wafer_to_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLUMN_MAPPING, SPEC_LIMITS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 資料載入、MES解析/清理、NPI流水線、Wafer Map矩陣轉換與特徵提取/分類
def main():
    logger.info("=== 半導體良率分析平台啟動 ===")
    
    # 1. 統一資料載入
    data = load_all_sources()
    
    # 2. MES Log 處理（專案一）
    mes_raw = parse_mes_logs(data["mes"])
    mes_clean = clean_mes_logs(mes_raw)
    
    # 3. NPI 良率分析（專案二）
    npi_result = run_npi_pipeline(data["npi"])

    # 4. Wafer Map 特徵提取與分類（專案三 + 四）
    wafer_matrices = [wafer_to_matrix(f) for f in data["wafer"]]
    wafer_features = extract_features(wafer_matrices)
    wafer_labels = classify_defects(wafer_features)
    
    # 5. 良率預測（專案五）
    prediction = predict_yield(
        mes_clean, npi_result, wafer_features, wafer_labels
    )
    
    
    # 6. 跨專案關聯分析與報告
    generate_report(prediction)
    
    logger.info("=== 分析與預測完成，報告已產出 ===")

if __name__ == "__main__":
    main()