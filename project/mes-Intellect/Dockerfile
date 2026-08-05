FROM python:3.10-slim

WORKDIR /app

# 安裝系統依賴（cv2 與編譯需求）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先複製依賴檔
COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling && pip install -e .[dev] || pip install -e .

# 複製其餘程式碼
COPY . .

# 預設啟動指令
CMD ["python", "-m", "src.main"]