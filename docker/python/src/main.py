import os
import time
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Python Ollama API", version="1.0.0")

# Ollamaクライアントの設定
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
client = ollama.Client(host=OLLAMA_BASE_URL)

class ChatRequest(BaseModel):
    model: str = "gemma3:4b"
    message: str
    stream: bool = False

class GenerateRequest(BaseModel):
    model: str = "gemma3:4b"
    prompt: str
    stream: bool = False

@app.get("/")
async def root():
    return {"message": "Python Ollama API is running!"}


@app.get("/health")
async def health_check():
    try:
        # Ollamaサービスの健全性チェック
        response = client.list()

        # モデル情報の取得
        models = []
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict) and "models" in response:
            models = response["models"]

        return {"status": "healthy", "ollama_connected": True, "models": models}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "ollama_connected": False}


@app.get("/models")
async def list_models():
    """利用可能なモデルリストを取得"""
    try:
        response = client.list()

        # 新しい構造に対応: "name" -> "model"
        if hasattr(response, 'models'):
            # response.models がある場合（新しい構造）
            models = response.models
        elif isinstance(response, dict) and "models" in response:
            # 辞書形式の場合（従来の構造）
            models = response["models"]
        else:
            # その他の場合
            models = []

        # モデル名の抽出（"model" キーを使用）
        model_names = []
        for model in models:
            if hasattr(model, 'model'):
                model_names.append(model.model)
            elif isinstance(model, dict) and "model" in model:
                model_names.append(model["model"])
            elif hasattr(model, 'name'):
                model_names.append(model.name)
            elif isinstance(model, dict) and "name" in model:
                model_names.append(model["name"])
            elif isinstance(model, str):
                model_names.append(model)

        return {"models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """チャット機能"""
    try:
        response = client.chat(
            model=request.model,
            messages=[
                {
                    'role': 'user',
                    'content': request.message,
                },
            ],
            stream=request.stream,
            options={"temperature": 0.7, "num_predict": 150, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 65536},
        )
        
        if request.stream:
            return {"message": "Streaming not implemented in this example"}
        else:
            return {
                "model": request.model,
                "message": response["message"]["content"],
                "done": response.get("done", True)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    """端的に答える"""
    try:
        response = client.generate(
            model=request.model,
            prompt=request.prompt,
            options={"temperature": 0.0, "num_predict": 30, "top_p": 0.3, "top_k": 10, "num_ctx": 65536},
            stream=request.stream
        )
        if request.stream:
            return {"message": "Streaming not implemented in this example"}
        else:
            return {
                "model": request.model,
                "response": response["response"],
                "done": response.get("done", True),
                "context": response.get("context", []),
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def wait_for_ollama():
    """Ollamaサービスが起動するまで待機"""
    print("Waiting for Ollama service to be ready...")
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            client.list()
            print("Ollama service is ready!")
            return True
        except Exception as e:
            retry_count += 1
            print(f"Retry {retry_count}/{max_retries}: Ollama not ready yet ({str(e)})")
            time.sleep(2)
    
    print("Failed to connect to Ollama service")
    return False

if __name__ == "__main__":
    # Ollamaサービスの起動を待つ
    if wait_for_ollama():
        print(f"Starting FastAPI server on http://0.0.0.0:8080")
        print(f"Ollama URL: {OLLAMA_BASE_URL}")
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        print("Could not connect to Ollama. Exiting.")
        exit(1)
