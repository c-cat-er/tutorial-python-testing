# 資料流與架構設計：繪製專案資料流程圖

- 目標：
    - 掌握 Pipeline 模組化設計：清晰定義系統從讀取設定、資料採集、清洗、多模組 AI 分析到最終寫入資料庫的單向流動架構。
    - 理解專案解耦（Decoupling）思維：確保每個資料夾與模組各司其職，避免產生牽一髮動全身的義大利麵條程式碼（Spaghetti Code）。

## 1. 專案核心資料流圖 (Data Flow Architecture)

- 整套自動化測試整合系統的資料流向為單向拓撲結構，由左至右、由上至下嚴格執行：
    -   1. 配置層：config：讀取 settings.yaml。
    -   2. 採集層：acquisition：掃描產線資料夾，讀取原始 MES Log / NPI 電性 CSV / 晶圓圖。
    -   3. 通用層：common：執行邏輯：
        - 正規表達式解析文字 (LOG_PATTERN)。
        - 欄位標準化對齊 (standardize_columns)。
    -   4. 核心 AI / 統計分析模組層
        - spc_charts：計算 X-bar R 與 Cpk/Ppk。
        - anomaly_detector：孤立森林與 AutoEncoder。
        - yield_predictor：XGBoost 預測良率。
        - wafer_classifier：ResNet18 晶圓缺陷分類。
    -   5. 整合與警報層：呼叫 4_3 SPC-ML 整合引擎，判定 alert_status。
    -   6. 輸出層：database/reporting：透過 3_1 SQLAlchemy ORM 寫入 DB。
        - 回填 YieldPredictionRecord。
        - 產出每日良率趨勢報表與看板。

## 2. 各模組核心權責與對應章節拆解

- 為了讓系統具備高度擴充性（例如未來要換掉預測模型或增加新機台），系統被高度解耦為以下模組：
    - 2.1 config (配置模組)
        - 職責：全專案的唯一指引。負責管理硬體 IP 位址、資料庫連接字串（Connection String）、產品 Spec 規格界限、以及機器學習模型決策門檻值。
        - 對應落實：settings.yaml。
    - 2.2 acquisition (資料採集模組)
        - 職責：負責硬體與系統對接。包含與測試探針床（Probe Card）的 SCPI 指令交握，以及使用 pd.read_csv(chunksize=...) 流式讀取產線巨量日誌，確保記憶體安全。
        - 對應落實：第一章的設備連線 try-except、第三章的 Chunk 讀取。
    - 2.3 common (通用清洗模組)
        - 職責：將無結構的髒資料標準化。利用正規表達式將文字日誌拆解為字典，並透過欄位映射表（Mapping）將異質欄位統一對齊，最後經由 Pydantic BaseModel 進行 Runtime 型別校驗。
        - 對應落實：第 1_4 節（Regex Parser）、第 3_2 節（欄位標準化）。
    - 2.4 各分析模組 (Core Analytics)
        - 各分析模組彼此獨立（解耦），主程式可以根據需求動態開啟或關閉特定分析：
            - mes_log_analyzer：統計各機台事件與當機頻率。
            - npi_yield_analyzer：利用 scipy.stats 執行跨機台 ANOVA 檢定與離群值剔除（第 2_2, 2_3 節）。
            - yield_predictor：投遞特徵給調優後的 XGBoost 模型，預測目標良率（第 4_1 節）。
            - wafer_map_classifier：將 2D 點位圖轉為 3D 張量，送入預訓練的 ResNet18 輸出缺陷分類結果（第 1_3, 5_1 節）。
    - 2.5 database / reporting (資料庫與報表輸出)
        - 職責：將分析與預測結果持久化。利用 SPCAlertEngine 計算出 alert_status（NORMAL / WARNING / CRITICAL）後，透過 ORM 一併寫入 SQLite/PostgreSQL 資料庫，並呼叫 Matplotlib 自動更新產線良率看板。
        - 對應落實：第 3_1 節（SQLAlchemy 建表）、第 4_3 節（Alert 整合）。

- 總結：在設計我的 GitHub 專案時，我引進了業界標準的軟體工程解耦思維，將整個自動化 Pipeline 劃分為六個清晰的單向流動層級。從 config 讀取參數，到 acquisition 透過 Chunk 機制安全採集，再到 common 利用 Regex 和 Pydantic 校驗守門，最後才分流投遞給獨立的分析模組（如 XGBoost 預測、ResNet18 分類、Cpk 統計計算）。這種 Pipeline 模組化設計 最大的物理好處在於：如果今天黃光 PE 想要微調缺陷分類模型（例如從 ResNet 換到 EfficientNet），他只需要改動 wafer_classifier 模組內部，完全不會影響到前端的日誌解析或後端資料庫的建表邏輯。這套兼具高強健性與可擴充性的數據架構，是確保 AI 模型能在半導體 24 小時產線真正落地運作的工程核心。
