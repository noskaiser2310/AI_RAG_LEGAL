import os
from huggingface_hub import snapshot_download

def download_models():
    # Thư mục lưu trữ model offline
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Danh sách model cần tải
    models = {
        "LLM": "Qwen/Qwen3-8B-Instruct",
        "Embedding": "mainguyen9/vietlegal-harrier-0.6b",
        "Reranker": "AITeamVN/Vietnamese_Reranker"
    }
    
    for name, repo_id in models.items():
        print(f"Downloading {name} model ({repo_id}) to {models_dir}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=os.path.join(models_dir, repo_id.replace("/", "_")),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "coreml/*"] # Bỏ qua các định dạng không cần thiết
        )
        print(f"✅ Downloaded {name} model successfully.")

if __name__ == "__main__":
    download_models()
