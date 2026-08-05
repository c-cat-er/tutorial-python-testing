本專案建立一套晶圓測試良率分析平台，整合 MES 製程紀錄、CTQ 品質參數與 Wafer Map 缺陷分布，透過缺陷型態辨識與 XGBoost 機器學習，找出影響晶圓良率的關鍵因素，並進行良率預測與異常追溯。

## 一、核心技術能力需求（四大類）

- A. 進階統計學與工業級資料視覺化
    - 核心技術：Tableau、PowerBI、JMP（半導體廠最愛）
    - 重點技術：ANOVA（方差分析）、假設檢定、SPC（統計製程管制）
    - 目標：從探針電性數據中快速判斷批次晶圓是否偏離正常範圍
- B. 電腦視覺缺陷分類（Wafer Map）
    - 核心技術：CNN（ResNet / EfficientNet）
    - 重點技術：Wafer Map Pattern Recognition
    - 目標：自動分類「邊緣失效」vs「刮痕失效」等缺陷型態，協助工程師定位問題機台
- C. 巨量資料庫與大數據處理
    - 核心技術：進階 SQL（Hadoop、Hive、Oracle）、Pandas 記憶體優化（pyarrow）
    - 重點：探針測試資料量達 Tera-byte 級，Excel 無法負荷，需高效查詢與清洗
- D. MLOps 模型防禦與自動化流水線
    - 核心技術：Docker、Airflow、CI/CD
    - 重點：Data Drift（資料漂移）監控 — 製程機台耗損會使模型準確率下降，須自動重訓練並上線
- 另附機器學習方法基礎
    - Python XGBoost：找出影響良率的關鍵特徵。
        - 結構化表格（電性測試參數、Bin Code 缺陷分類等）
        - 用樹模型（XGBoost / LightGBM）來找出哪些電性指標（特徵）最會影響良率，是基本日常。
    - OpenCV / CNN：自動偵測 Wafer Map 刮痕與異常。
    - K-Fold 交叉驗證 + 回歸/分類預測：預測良率、分類缺陷型態。

## 二、專案架構：Semiconductor Yield Integration Platform

- 整合五個子專案的優化版本：
    - MES Log Cleaner
    - NPI CTQ Yield Board
    - Bin Map Pattern Recognition（專案三＋四）
    - Lot Yield Predictor（專案五）
- 資料分層流程：raw → bronze（清洗） → silver（特徵＋統計） → gold（模型輸出＋報表）
    - Raw：MES Log、NPI / CTQ Data、Wafer Map。
    - silver：清洗 + 特徵工程、Wafer Map 分析、CTQ 統計分析。
    - gold：XGBoost 良率預測、缺陷分類、報表與警示。
- Traceability（可追溯性)：所有資料表必須包含：lot_id、wafer_id、die_id。

## 三、自動化數據流水線（執行 docker-compose up 或 python src/main.py 後）

| 階段               | 模組                       | 功能                                                                      |
| :----------------- | :------------------------- | :------------------------------------------------------------------------ |
| **Bronze**         | `loader.py`                | 自動搜尋 `data/raw/` 下的 **MES 日誌**、**NPI CSV**、**Wafer Map** 數據。 |
| **Silver（清洗）** | `parser.py` + `cleaner.py` | 解析並計算 **MES Log 嚴重程度（severity）**。                             |
| **Silver（清洗）** | `pipeline.py`              | 對齊 **NPI 欄位**、計算 **CTQ 指標**。                                    |
| **Silver（清洗）** | `wafer_to_matrix()`        | 將 **晶圓地圖轉為二值矩陣**。                                             |
| **Silver（特徵）** | `feature_extractor.py`     | 計算 **缺陷佔比**、**邊緣佔比**、**輪廓獨立個數**。                       |
| **Silver（特徵）** | `classifier.py`            | **缺陷型態分類**（如 Ring、Random 等）。                                  |
| **Gold**           | `predictor.py`             | 整合三大模組特徵，用 **XGBoost 預測良率**。                               |
| **Gold**           | `reporting.py`             | 自動產出 **整合分析報告與警示檔案**。                                     |

- **⚡ 主流程核心執行順序規範**：
  `載入資料` $\rightarrow$ `MES Log 解析清理` $\rightarrow$ `NPI 流水線` $\rightarrow$ `Wafer Map 特徵提取與分類` $\rightarrow$ `XGBoost 良率預測` $\rightarrow$ `產出報告`。

## 四、資料格式要求

1. MES Log（data/raw/mes_logs/\*.log）
    - MES 生產紀錄，分析 Lot、Wafer、Equipment、Process Step、Timestamp、Event Severity。
    - 純文字 UTF-8，每行一筆事件，用 | 或 , 分隔。
    - 必備欄位：lot_id、wafer_id、timestamp（ISO格式）、equipment_id、process_step
2. NPI / CTQ 良率資料（data/raw/npi/）
    - CTQ = Critical to Quality 關鍵品質參數，分析 CD、Thickness、Overlay、Electrical Parameters、Yield %。
    - CSV（逗號分隔＋標題列）或固定寬度文字檔
    - 推薦欄位：lot_id、wafer_id、sample_id、ctq_parameter（如 CD、Thickness、Overlay）、value、unit、yield_pct。
    - 可以搭配：統計分析、Regression、ANOVA、Hypothesis Testing、SPC。
3. Wafer Map 資料（data/raw/wafer_maps/\*.csv）
    - 建議統一使用「稀疏座標格式」
    - 必備欄位：lot_id、wafer_id、die_x（或 col）、die_y（或 row）、bin_code（0=Pass，1~N=各類 Fail）
    - 每一個 Die 都有：die_x、die_y、bin_code。

## 、主要分析的缺陷型態

- 🔵 Ring Pattern：晶圓邊緣或環狀區域大量失效。
    - 可能代表：製程邊緣效應、設備均勻性問題。
- 🔴 Scratch Pattern：呈現線狀或刮痕型缺陷。
    - 可能代表：搬運、設備或製程機械問題。
- 🟡 Random Pattern：缺陷隨機分布。
    - 可能代表：隨機污染或偶發性製程問題。
- 其他可分析：Edge Failure、Center Failure、Cluster Failure、Local Defect。
- 系統會計算：缺陷比例、邊緣缺陷比例、缺陷輪廓數量、缺陷幾何分布。再將晶圓分類為 Ring、Random 等不同缺陷類型。

## 、XGBoost：預測晶圓良率 (AI 核心)

- XGBoost 整合：MES 異常程度、CTQ 品質參數、Wafer Map 缺陷比例、Wafer Map Pattern、Bin Code 分布、Electrical Test Parameters。
- 輸出結果：預測這片晶圓的 Yield 可能是多少，以及哪些特徵最影響良率？

## 五、啟動方式

方法一：Docker
bash
mkdir -p data logs
docker-compose up --build -d # --build 強制重建映像；-d 背景執行
docker-compose logs -f # 查看日誌
docker-compose down # 停止服務
方法二：本地環境
bash
mkdir -p data/raw/mes_logs data/raw/npi data/raw/wafer_maps data/gold/models logs
cp .env.example .env # 設定 DATABASE_URL、模型路徑等
pip install -e . # 安裝 pandas, numpy, xgboost, scikit-learn, # opencv-python-headless, sqlalchemy, pydantic 等
python src/main.py # 或 python -m yield_platform.main --mode full
主流程順序：載入資料 → MES Log 解析清理 → NPI 流水線 → Wafer Map 特徵提取與分類 → XGBoost 良率預測 → 產出報告

## 六、執行後成果檢查點 ⚠️（勿漏）

- 資料庫：data/gold/yield.db（SQLite）— 清洗分析後的結構化數據
- 模型檔：data/gold/models/yield_predictor.pkl
- 系統日誌：logs/app.log

- 備註：目前設定僅啟動單一 app 服務；若需 DB、worker 等其他服務，須自行擴充 docker-compose.yml。
