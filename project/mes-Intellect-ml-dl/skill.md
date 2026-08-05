# 專案具體使用技術

## 🛠️ 1. 開發環境與專案管理

- 負責專案的<my-orange>運作環境、套件版本控制與容器化部署</my-orange>。

1. Python 3.10+。
2. pyproject.toml：現代 Python 的專案設定檔，用來管理專案套件的依賴版本。
3. Docker + docker-compose：將程式和運作環境<my-orange>打包</my-orange>。
4. python:3.10-slim：<my-orange>輕量化的 Linux 基礎映像檔，用來降低 Docker 檔案體積</my-orange>。

## 📦 2. 資料庫與資料持久化

- 負責儲存、讀取與管理專案的結構化資料。

1. SQLite：輕量級的嵌入式資料庫，不需要獨立安裝伺服器，檔案直接存在本機。
2. SQLAlchemy ORM：將資料庫欄位對應成 Python 的物件（物件關係對應），讓工程師不用寫原生 SQL 語法就能操作資料庫。

## 🧹 3. 資料清洗與日誌解析

- 將原始、雜亂的文字與數據，轉換成結構化、乾淨的資料。

1. 正規表達式 (<my-cyan>Regex</my-cyan>)：用來篩選、抓取製造執行系統（MES）日誌（Logs）中的特定關鍵字。
2. LabelEncoder：將類別文字（例如：機台代碼 A/B/C）轉換成數字（0/1/2），以便電腦計算。
3. pivot_table (<my-cyan>樞紐分析表</my-cyan>)：將一維的流水帳資料，重組為二維的矩陣（在這裡用於重現 wafer map 晶圓地圖的幾何分佈）。
4. <my-cyan>pandas, numpy, pyarrow, openpyxl</my-cyan>：核心的<my-orange>數據處理與 Excel 檔案讀寫工具</my-orange>。

## 👁️ 4. 影像處理與特徵提取

- 處理半導體晶圓地圖（Wafer Map）等圖形資料，並從中萃取出可用的特徵指標。

1. <my-cyan>OpenCV</my-cyan> (opencv-python-headless)：電腦視覺庫（headless 版本代表不含 GUI 畫面，適合伺服器執行）。
2. <my-cyan>Canny 邊緣檢測</my-cyan>：<my-orange>自動找出</my-orange>晶圓影像或圖形中的<my-orange>邊界、線條與異常形狀</my-orange>。
3. scipy.stats：<my-orange>計算統計特徵（如偏態、峰度），將影像特徵轉化為數值</my-orange>。

## 🤖 5. 機器學習與模型解釋

- 利用處理好的特徵建立預測模型，並解釋模型的決策原因。

1. <my-cyan>XGBoost (scikit-learn)</my-cyan>：<my-orange>強大的梯度提升樹演算法，用於預測晶圓的良率或分類缺陷</my-orange>。
2. <my-cyan>SHAP</my-cyan>：機器學習<my-orange>解釋工具</my-orange>。因為 AI 模型常被視為黑盒子，SHAP 可以告訴你「為什麼模型預測這片晶圓會失敗？是哪個特徵影響最大？」。

## 📊 6. 系統配置、日誌與資料視覺化

- 負責專案的系統設定、錯誤紀錄與圖表繪製。

1. YAML (pyyaml)：可讀性高的設定檔格式。
    - settings.yaml：管理專案全局設定（如資料庫路徑、模型參數）。
    - logging.yaml：管理系統日誌的輸出格式與儲存路徑。
2. pydantic：嚴格驗證資料格式是否正確（例如確保設定檔中的數字不會被誤填成文字）。
3. <my-cyan>matplotlib, seaborn</my-cyan>：繪圖工具，用來畫出良率趨勢圖、缺陷分佈圖或 SHAP 解釋圖。

# 缺少的具體技術

- XGBoost 模型訓練/預測 + SHAP 解釋 (predictor.py)
- sklearn 完整分類器或規則分類 (classifier.py)
- 儀器通訊整合 (hardware.py, pyvisa/pyserial)
- 空間幾何特徵 (geometry.py, shapely/scipy.spatial)
- 進階統計分析 (statistical.py, statsmodels)
- logging.yaml 完整日誌配置
- probe_configs/default.yaml + 探針卡規格
- Pydantic 完整資料驗證模型
- matplotlib/seaborn 視覺化實作
- SQLAlchemy 完整 ORM 模型定義

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 粉紅，錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 橘，警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 綠，正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 藍，提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 灰，次要註解 */
</style>
