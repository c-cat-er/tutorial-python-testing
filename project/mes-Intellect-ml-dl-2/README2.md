未整合到 README.md

# Semiconductor Yield Integration Platform (小型整合框架)

整合五個專案的優化版本：

- MES Log Cleaner
- NPI CTQ Yield Board
- Bin Map Pattern Recognition (專案三 + 四)
- Lot Yield Predictor (專案五)

## 資料流程

raw → bronze（清洗） → silver（特徵 + 統計） → gold（模型輸出 + 報表）。

## Traceability

所有資料表皆包含 `lot_id`, `wafer_id`, `die_id`。

## 完整的自動化數據流 (Data Pipeline)

現在當你執行 docker-compose up 或本地執行 python src/main.py 時，程式會自動完成以下全套閉環流程：

1. 資料讀取 (Bronze Layer)：loader.py 自動至 data/raw/ 搜尋 MES 日誌、NPI CSV 與 Wafer Map 數據。
2. 清洗與結構化 (Silver Layer)：

- parser.py + cleaner.py 解析並計算 MES Log 的嚴重程度等級 (severity)。
- pipeline.py 對齊 NPI 欄位並計算關鍵 CTQ 指標。
- wafer_to_matrix() 將晶圓地圖轉為二值矩陣。

3. 特徵提取與模式識別 (Silver Layer)：

- feature_extractor.py 計算缺陷佔比、邊緣佔比與輪廓獨立個數。
- classifier.py 進行缺陷型態分類（如 Ring, Random 等）。

4. 預測與報告產出 (Gold Layer)：

- predictor.py 整合三大模組特徵，餵入 XGBoost 模型預測良率。
- reporting.py 自動產出整合分析報告與警示檔案。

## 快速啟動

### 虛擬環境執行

1. ctrl + shift + p，輸入 Python: Select Interpreter，選擇創建虛擬環境。
2. 強迫 Python 在初始化載入（init_import_site）時，一律使用 UTF-8 讀取所有套件路徑檔案。

```bash
$env:PYTHONUTF8=1
```

3. 開啟終端機，輸入以下安裝

- 執行重新確認專案內部的所有依賴。

```bash
$env:PYTHONUTF8=1
pip install . --force-reinstall --no-deps
pip install .
```

```bash
pip install hatchling
```

- 安裝 XGBoost。

```bash
pip install xgboost shap scikit-learn
```

4. 生成原始資料

```bash
python generate_mock_data.py
```

5. 跑平台分析

```bash #(PowerShell)
python scripts/run_orchestrator.py
```

6. 若出現 CP950 錯誤，則刪除 venv/Lib/site-packages/\_editable_impl_semiconductor_yield_platform.pth 檔案。

7. 清除所有 **pycache** 子資料夾

```bash #(PowerShell)
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Force -Recurse
```

### 方法一：使用 Docker 容器化執行

1. 建立必要資料夾 `mkdir -p data logs`。

- 放入原始資料
    - 1-1. MES Log 檔案：放入 data/raw/mes_logs/ (例如 \*.log 檔)
        - 檔案格式：純文字日誌（UTF-8），每行一筆事件記錄，建議使用 | 或 , 分隔欄位。
        - 必備欄位：
            - [lot_id]、[wafer_id]（ traceability 核心）
            - [timestamp]（ISO 格式或 yyyy-mm-dd HH:MM:SS）
            - [equipment_id]、[process_step]
    - 1-2. NPI 良率資料：放入 data/raw/npi/ (例如 _.csv 或 _.txt 檔)
        - 檔案格式：CSV（推薦使用逗號分隔，含標題列），或固定寬度文字檔。
        - 推薦欄位（依 NPI CTQ Yield Board 需求）：
            - [lot_id]、[wafer_id]、[sample_id]
            - [ctq_parameter]（關鍵品質參數，例如 CD、Thickness、Overlay 等）
            - [value]、[unit]、[yield_pct]
    - 1-3. Wafer Map 資料：放入 data/raw/wafer_maps/ (例如 \*.csv 檔)
        - 檔案格式：兩種常見正規樣式（建議統一使用「稀疏座標格式」以利後續處理）。
        - 必備欄位：
            - [lot_id]、[wafer_id]、[die_x]（或 col）、die_y（或 row）
            - [bin_code]（0=Pass、1~N=各類 Fail bin）

2. 一鍵建置並在背景執行 `docker-compose up --build -d`。
    - 首次執行時，Docker 會自動讀取 Dockerfile 補捉 `pyproject.toml` 內的依賴，下載鏡像並完成編譯。
    - --build：強制重新建置映像檔（確保程式碼或套件有更新時能同步）。
    - -d：在背景執行（Detached mode），放開您的終端機。
3. 查看執行日誌與狀態 `docker-compose logs -f`。
4. 停止服務 `docker-compose down`。

### 方法二：本地環境直接執行

1. 初始化環境建立必要資料夾 `mkdir -p data logs`。

```bash
mkdir -p data/raw/mes_logs data/raw/npi data/raw/wafer_maps data/gold/models logs
```

2. 複製環境變數設定 `cp .env.example .env`
   將環境變數範本 .env.example 複製一份並命名為 .env
   根據需求打開 .env 檢查或修改資料庫路徑 DATABASE_URL 或模型路徑。
3. 安裝專案依賴套件 `pip install -e .`
   會自動讀取 `pyproject.toml` 並安裝 pandas, numpy, xgboost, scikit-learn, opencv-python-headless, sqlalchemy, pydantic 等所有必要套件。
4. 放入原始資料
    - MES Log 檔案：放入 data/raw/mes*logs/ (例如 *.log 檔)。
    - NPI 良率資料：放入 data/raw/npi/ (例如 _.csv 或 _.txt 檔)。
    - Wafer Map 資料：放入 data/raw/wafer*maps/ (例如 *.csv 檔)。
5. 執行主程式 `python src/main.py` 或 `python -m yield_platform.main --mode full`。
   (主程式會依序啟動：統一載入資料 -> MES Log 解析與清理 -> NPI 流水線 -> Wafer Map 特徵提取與缺陷分類 -> XGBoost 良率預測 -> 自動產出分析與警示報告。)
6. 查看日誌 `docker-compose logs -f`。

## 💡 執行後的成果檢查點

- 資料庫更新：檢查 `data/gold/yield.db（SQLite`，清洗與分析後的結構化數據會被寫入其中。
- 模型儲存：預測模型會產出並儲存在 `data/gold/models/yield_predictor.pkl`。
- 系統日誌：產出的執行細節會紀錄在 `logs/app.log`。

目前設定只啟動單一 app 服務。若有其他服務（DB、worker 等），需再擴充 `docker-compose.yml`。

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 粉紅，錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 橘，警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 綠，正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 藍，提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 灰，次要註解 */
</style>
