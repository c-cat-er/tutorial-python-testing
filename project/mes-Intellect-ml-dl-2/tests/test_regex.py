import re

# 1. 複製 Page 15-16 專案原生的正規表示式
LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|"
    r"(?P<lot_id>[^|]+)\|"
    r"(?P<wafer_id>[^|]+)\|"
    r"(?P<equipment_id>[^|]+)\|"
    r"(?P<process_step>[^|]+)\|"
    r"(?P<event_type>[^|]+)\|"
    r"(?P<message>[^|]+)\|"
    r"(?P<operator>[^|]+)\|"
    r"(?P<status>[^|\n]+)"
)

# 2. 複製 Page 37 實體產線產出的標準日誌文本
sample_line = "2026-07-20 08:15:23|LOT20260720001|W01|EQ-LITHO-03|LITHOGRAPHY|START|Process started|OP-001|RUNNING"

# 3. 執行檢測
match = LOG_PATTERN.search(sample_line)

print("=== 產線 Regex 擷取欄位交叉對齊檢測 ===")
if match:
    parsed_dict = match.groupdict()
    for key, value in parsed_dict.items():
        print(f"欄位 [{key:12}] -> 解析結果: {value}")
else:
    print("❌ 錯誤：Regex 完全無法匹配該行日誌，請檢查分隔符號或特殊字元！")
