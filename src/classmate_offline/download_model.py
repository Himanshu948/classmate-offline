from huggingface_hub import snapshot_download

from classmate_offline.asr import MODEL_ID
from classmate_offline.paths import WHISPER_MODEL_DIR


def main() -> None:
    WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(WHISPER_MODEL_DIR),
    )
    print(f"Model downloaded to {WHISPER_MODEL_DIR}")


if __name__ == "__main__":
    main()