## 專案架構介紹

### 環境與資料夾結構

- data/raw/ 內的 MES logs、NPI CSV/TXT、Wafer Map CSV 等範例資料格式均已定義完備。
- .env.example、pyproject.toml、docker-compose.yml 與 config/settings.yaml 提供完美的環境配置與模組依賴關係。

### 核心子模組功能

- 資料載入與解析：loader.py、parser.py、cleaner.py、pipeline.py。
- 特徵提取與建模：feature_extractor.py、classifier.py、predictor.py。
- 報告邏輯：reporting.py。

### 還缺

主程式主控台 (src/main.py) 僅撰寫了引用 (import) 與印出標題，尚未將變數與函式串接：
需要補齊：呼叫 load_all_sources() -> 執行 parse_mes_logs / clean_mes_logs -> 執行 run_npi_pipeline -> 轉譯 Wafer Map 陣列並呼叫 extract_features 與 classify_defects -> 將特徵組合傳入 predict_yield -> 呼叫 generate_report。

Wafer Map CSV 轉 2D 陣列：loader.py 讀進來的是原始路徑，feature_extractor.py 的 extract_features(wafer_maps) 預期接收的是 2D/3D Numpy Array (bin_map) 。中間需要有一小段「將 Wafer Map CSV 的 die_x, die_y, bin_code 轉為矩陣」的 Pivot 處理。

Predictor 輸入格式相容：predict_yield(mes_df, npi_df, wafer_features, wafer_labels) 內部使用了 pd.concat([mes_df[['severity']], npi_df[['Value']], ...], axis=1)。
注意 npi_df 在 pipeline.py 清理後欄位大小可能與 mes_df 或 wafer_features 列數 (Row length) 不一致，直接 axis=1 合併需要確保過濾/對齊 lot_id 與 wafer_id。
