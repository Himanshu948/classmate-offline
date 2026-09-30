from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHISPER_MODEL_DIR = PROJECT_ROOT / "models" / "faster-whisper-small"
LLM_MODEL_DIR = PROJECT_ROOT / "models" / "llm"