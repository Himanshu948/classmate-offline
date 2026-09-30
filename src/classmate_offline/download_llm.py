from gpt4all import GPT4All

from classmate_offline.paths import LLM_MODEL_DIR
from classmate_offline.study_pack import LLM_MODEL_NAME


def main() -> None:
    LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GPT4All.retrieve_model(
        LLM_MODEL_NAME,
        model_path=str(LLM_MODEL_DIR),
        allow_download=True,
        verbose=True,
    )
    print(f"Model downloaded to {LLM_MODEL_DIR / LLM_MODEL_NAME}")


if __name__ == "__main__":
    main()