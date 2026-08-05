import pandas as pd
from datetime import datetime
from analysis.reporting import generate_report
from yield_predictor.predictor import predict_yield

def main():
    # 假設已從前一步驟取得資料
    prediction = predict_yield(...)  # 實際使用時傳入真實資料
    generate_report(prediction)
    
    print(f"[{datetime.now()}] 自動報告已產生至 data/gold/report.csv")

if __name__ == "__main__":
    main()