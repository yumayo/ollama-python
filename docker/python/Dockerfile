FROM python:3.13.3-slim

WORKDIR /app/src

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY docker/python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY docker/python/src/ .

EXPOSE 8000

CMD ["python", "main.py"]
