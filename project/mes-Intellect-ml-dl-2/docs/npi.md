## NPI 簡述

- <my-orange>NPI 代表新產品導入（New Product Introduction），CTQ 代表關鍵品質特性 (Critical to Quality)</my-orange>。這是工程師針對「整批產品」進行的抽樣點檢表、量測統計值 (如平均值、標準差、良率看板總結)。
- 現代 NPI 階段會特別關注 CD（線寬）、THICKNESS（膜厚）、OVERLAY（疊對誤差） 等幾何量測特徵，這些數據通常來自光學量測機台（Metrology）。
- 內含 sample_id（抽樣點點號），因為量測機台速度慢、成本高，不可能像測試機台那樣全檢，通常在一片晶圓上只抽樣量測固定幾個點（如 9 點、13 點測試）。

## 檔案欄位說明

| lot_id         | wafer_id | sample_id  | ctq_parameter                           | value      |
| -------------- | -------- | ---------- | --------------------------------------- | ---------- |
| 批次編號       | 晶圓編號 | 樣本編號   | 關鍵品質特性參數                        | 實際量測值 |
| LOT20260720001 | W01、W02 | S001、S002 | 線寬 CD_LINEWIDTH、THICKNESS、OVERLAY_X | ---        |

| unit     | measurement_time    | yield_pct  | operator     |
| -------- | ------------------- | ---------- | ------------ |
| 量測單位 | 量測時間點          | 良率百分比 | 操作人員代號 |
| um、nm   | 2026-07-20 09:15:00 | ---        | OP-001       |

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
