# 修改 src/analysis/reporting.py
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.utils import ensure_dir


def generate_report(prediction_result):
    """產出 CSV 報告並自動繪製工業級分析圖表"""
    ensure_dir("data/gold")

    # 1. 產生原本的 CSV 紀錄
    report = {
        "timestamp": datetime.now().isoformat(),
        "predicted_yield": prediction_result["predicted_yield"],
        "alert": "高風險" if prediction_result["predicted_yield"] < 90.0 else "正常",
    }
    pd.DataFrame([report]).to_csv("data/gold/report.csv", index=False)

    # 2. 補齊 Matplotlib / Seaborn 視覺化實作
    factors = prediction_result["top_factors"]
    df_plot = pd.DataFrame(factors, columns=["Feature", "Importance"])

    plt.figure(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=df_plot, palette="viridis")
    plt.title("Top Factors Affecting Yield (SHAP Analysis)")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()

    # 保存圖表，供後續展示或看板讀取
    plt.savefig("data/gold/yield_factors_chart.png", dpi=300)
    plt.close()
    print("報告與分析圖表已生成至 data/gold/")
