import sys
from pathlib import Path

# 取得專案根目錄的絕對路徑
project_root = Path(__file__).resolve().parent.parent

# 乾淨地將專案路徑加入 Python 搜尋路徑
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

# 自動建立持久化資料夾
(project_root / "data" / "gold" / "models").mkdir(parents=True, exist_ok=True)

from src.main import main

if __name__ == "__main__":
    main()
