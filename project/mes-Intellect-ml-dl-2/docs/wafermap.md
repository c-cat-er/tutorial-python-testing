## wafer map 簡述

- 檔案名 W01 指第一片。

## 常見兩種檔案型式

1. <my-cyan>稀疏座標格式</my-cyan> (推薦，適合<my-orange>大尺寸晶圓</my-orange>): `wafermap_F12_20260720_LOT001_W01 copy.csv`

- 座標制：只記錄有被測試到的 Die 座標，這在大晶圓（動輒上萬顆 Die）中能極大化節省儲存空間。
- 智能特徵：包含了 defect_type（如 Scratch 刮傷、Ring 環狀缺陷）。現代 12 吋廠的 ADC（自動缺陷分類）系統會在測試完後，自動透過 AI 電腦視覺演算法為晶圓圖打上缺陷標籤，並記錄精確的 test_time（測試時間戳記）。

2. <my-cyan>密集矩陣格式</my-cyan> (<my-orange>較小晶圓或完整 map</my-orange>): `wafermap_F8_20260720_LOT001_W01.csv`

- 矩陣/陣列制：老廠系統（如 8 吋或 6 吋）由於機台古老，通常不輸出實體座標，而是直接輸出一個棋盤格矩陣（Row 0, Col 0 -> Row 0, Col 1）。
- 資料相對單純，通常只有 bin_code（0代表Pass，1代表Fail），不帶有進階的 AI 缺陷分類或時間戳記。

## 檔案欄位說明 F12

| lot_id         | wafer_id | die_x             | die_y             |
| -------------- | -------- | ----------------- | ----------------- |
| 批次編號       | 晶圓編號 | 晶粒 X 軸水平位置 | 晶粒 Y 軸垂直位置 |
| LOT20260720001 | W01、W02 | ---               | ---               |

| bin_code | defect_type         | test_time           |
| -------- | ------------------- | ------------------- |
| 分類代碼 | 缺陷類型            | 量測時間點          |
| ---      | Pass、Scratch、Ring | 2026-07-20 10:05:01 |

## 檔案欄位說明 F8

| lot_id         | wafer_id | row               | col               | bin_code |
| -------------- | -------- | ----------------- | ----------------- | -------- |
| 批次編號       | 晶圓編號 | 晶粒列座標 Y 位置 | 晶粒行座標 X 位置 | 分類代碼 |
| LOT20260720001 | W01、W02 | ---               | ---               | ---      |

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
