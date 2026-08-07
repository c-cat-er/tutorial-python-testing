import pandas as pd
import scipy.stats as stats


def correlate_probe_final_test(npi_df):
    """CP 探針測試 (Probe) 與 FT 最終測試 (Final Test) 的跨產線資料關聯分析"""
    # 確保關鍵特徵欄位沒有空值，避免統計檢定崩潰
    clean_df = npi_df[["value", "yield_pct"]].dropna()

    # 樣本數太少時防禦性回傳（Pearsonr 最少需要 2 筆，工業級建議至少 5 筆）
    if len(clean_df) < 5:
        return {
            "status": "Low_Sample",
            "pearson_r": 0.0,
            "p_value": 1.0,
            "is_statistically_significant": False,
            "correlation_alert": "Low",
            "action_required": "資料樣本不足",
        }

    # 計算 CP 電性特徵 (value) 與 FT 最終良率 (yield_pct) 的相關係數
    r_coef, p_value = stats.pearsonr(clean_df["value"], clean_df["yield_pct"])

    # 工業級統計判讀邏輯（融合兩段邏輯，設定顯著性 P-value 門檻）
    is_significant = p_value < 0.05
    # 達到統計顯著（P < 0.05）且高度相關（|r| > 0.5）才發出高風險警示
    alert_level = "High" if (is_significant and abs(r_coef) > 0.5) else "Low"

    # 統一格式化輸出（不重複 return）
    return {
        "status": "Success",
        "pearson_r": float(r_coef),
        "p_value": float(p_value),
        "is_statistically_significant": bool(is_significant),
        "correlation_alert": alert_level,
        "action_required": "建議檢查前段機台參數"
        if alert_level == "High"
        else "正常監控",
    }
