import pandas as pd
from datetime import datetime

def generate_report(prediction):
    report = {
        "timestamp": datetime.now().isoformat(),
        "predicted_yield": prediction["predicted_yield"],
        "alert": "高風險" if prediction["predicted_yield"] < 0.9 else "正常"
    }
    pd.DataFrame([report]).to_csv("data/gold/report.csv", index=False)
    print("報告已生成")