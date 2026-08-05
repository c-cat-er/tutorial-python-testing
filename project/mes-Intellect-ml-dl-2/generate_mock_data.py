import os
import random
from datetime import datetime, timedelta
from pathlib import Path


def create_mock_data():
    print("=== 開始生成 100% 復刻生產線格式的三天份原始資料 ===")

    base_paths = ["data/raw/mes_logs", "data/raw/npi", "data/raw/wafer_maps", "logs"]
    for p in base_paths:
        Path(p).mkdir(parents=True, exist_ok=True)

    dates = ["2026-07-20", "2026-07-21", "2026-07-22"]
    lots = ["LOT20260720001", "LOT20260721002", "LOT20260722003"]
    wafers = ["W01", "W02"]

    # 2. 生成 100% 符合說明書 Page 38 範例的實體 .log 檔案
    for i, date_str in enumerate(dates):
        lot_id = lots[i]
        log_file = Path(f"data/raw/mes_logs/mes_{date_str.replace('-', '')}_LOT001.log")
        has_anomaly = date_str == "2026-07-22"

        with open(log_file, "w", encoding="utf-8") as f:
            # 寫入神聖的產線標頭列
            f.write(
                "timestamp|lot_id|wafer_id|equipment_id|process_step|event_type|message|operator|status\n"
            )

            for w_id in wafers:
                base_time = datetime.strptime(
                    f"{date_str} 08:00:00", "%Y-%m-%d %H:%M:%S"
                )

                lines = [
                    (0, "INFO", "LITHOGRAPHY", "START", "Process started"),
                    (2, "INFO", "LITHOGRAPHY", "INFO", "Recipe loaded: RECIPE-V2.3"),
                ]

                if has_anomaly and w_id == "W01":
                    lines.append(
                        (
                            7,
                            "CRITICAL",
                            "LITHOGRAPHY",
                            "WARNING",
                            "Overlay shift detected: 0.12um",
                        )
                    )
                else:
                    lines.append(
                        (
                            7,
                            "INFO",
                            "LITHOGRAPHY",
                            "INFO",
                            "Overlay check: 0.02um Passed",
                        )
                    )

                lines.extend(
                    [
                        (15, "INFO", "LITHOGRAPHY", "END", "Process completed"),
                        (20, "INFO", "ETCHING", "START", "Process started"),
                        (40, "INFO", "ETCHING", "END", "Process completed"),
                    ]
                )

                for min_offset, lvl, proc, ev_type, msg in lines:
                    t = (base_time + timedelta(minutes=min_offset)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    eq = "EQ-LITHO-03" if proc == "LITHOGRAPHY" else "EQ-ETCH-07"
                    op = "OP-001" if proc == "LITHOGRAPHY" else "OP-002"
                    stat = "RUNNING" if ev_type != "END" else "COMPLETED"

                    f.write(
                        f"{t}|{lot_id}|{w_id}|{eq}|{proc}|{ev_type}|{msg}|{op}|{stat}\n"
                    )

        print(f"  [已生成標準 MES Log] -> {log_file}")

    # 3. 生成 NPI 良率資料 (精準對齊說明書)
    for i, date_str in enumerate(dates):
        lot_id = lots[i]
        npi_file = Path(f"data/raw/npi/npi_ctq_{date_str.replace('-', '')}_LOT001.csv")
        has_anomaly = date_str == "2026-07-22"

        with open(npi_file, "w", encoding="utf-8") as f:
            f.write(
                "lot_id,wafer_id,sample_id,ctq_parameter,value,unit,measurement_time,yield_pct,operator\n"
            )
            w01_yield = 87.2 if has_anomaly else round(random.uniform(97.5, 99.5), 1)
            w01_cd = 0.092 if has_anomaly else 0.085
            f.write(
                f"{lot_id},W01,S001,CD_LINEWIDTH,{w01_cd},um,{date_str} 09:15:00,{w01_yield},OP-003\n"
            )
            f.write(
                f"{lot_id},W01,S002,THICKNESS,245.3,nm,{date_str} 09:16:12,{w01_yield},OP-003\n"
            )
            w02_yield = round(random.uniform(97.5, 99.5), 1)
            f.write(
                f"{lot_id},W02,S001,CD_LINEWIDTH,0.082,um,{date_str} 09:18:45,{w02_yield},OP-003\n"
            )
            f.write(
                f"{lot_id},W02,S003,OVERLAY_X,0.015,um,{date_str} 09:20:33,{w02_yield},OP-004\n"
            )
        print(f"  [已生成合格 NPI CSV] -> {npi_file}")

    # 4. 生成 Wafer Map 資料 (F12 稀疏座標格式)
    radius = 14
    center = 15.5
    for i, date_str in enumerate(dates):
        lot_id = lots[i]
        has_anomaly = date_str == "2026-07-22"
        for w_id in wafers:
            wm_file = Path(
                f"data/raw/wafer_maps/wafermap_F12_{date_str.replace('-', '')}_LOT001_{w_id}.csv"
            )
            with open(wm_file, "w", encoding="utf-8") as f:
                f.write("lot_id,wafer_id,die_x,die_y,bin_code,defect_type,test_time\n")
                for y in range(32):
                    for x in range(32):
                        if ((x - center) ** 2 + (y - center) ** 2) <= radius**2:
                            bin_code = 1
                            defect_type = "Pass"
                            if has_anomaly and w_id == "W01":
                                dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                                if 10.0 <= dist <= 12.5:
                                    if random.random() < 0.75:
                                        bin_code = 2
                                        defect_type = "Ring"
                            if bin_code == 1 and random.random() < 0.005:
                                bin_code = 4
                                defect_type = "Random"
                            f.write(
                                f"{lot_id},{w_id},{x},{y},{bin_code},{defect_type},{date_str} 10:05:00\n"
                            )
            print(f"  [已生成合格 WaferMap] -> {wm_file}")

    print("=== 生產線標準原始資料刷新完畢！ ===")


if __name__ == "__main__":
    create_mock_data()
