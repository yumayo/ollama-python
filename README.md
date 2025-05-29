念の為、ollamaを外部ネットワークから隔離した状態で動かす

# モデルのダウンロード
docker compose exec ollama-downloader ollama pull gemma3:4b

# モデルのコピー
docker compose run ollama-sync rsync -av --progress /app/ollama/download/.ollama/ /app/ollama/internal/.ollama/

# モデルの削除
docker compose exec ollama-downloader rm -rf /app/ollama/download/.ollama

# モデルの一覧取得
curl -X GET http://localhost:8000/models

# チャット
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"model": "gemma3:4b", "message": "Hello, how are you?"}'

# 端的な内容
curl -X POST http://localhost:8080/generate -H "Content-Type: application/json" -d '{"model": "gemma3:4b", "prompt": "Answer briefly and concisely: Explain REST API"}'

