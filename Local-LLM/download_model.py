from pathlib import Path
from huggingface_hub import snapshot_download


def download_quantized_model():

    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    repos_to_try = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"

    try:
        print(f"量子化モデルをダウンロード中: {repos_to_try}")
        print(f"ダウンロード先: {models_path.absolute()}")

        snapshot_download(
            repo_id=repos_to_try,
            allow_patterns=["*Q3_K_M.gguf"],
            local_dir=models_path,
            local_dir_use_symlinks=False,
        )
        print("ダウンロード完了！")
    except (OSError, ValueError, RuntimeError):
        print("ダウンロードに失敗しました。")


if __name__ == "__main__":
    download_quantized_model()
