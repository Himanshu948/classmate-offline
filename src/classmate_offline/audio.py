import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
SUPPORTED_AUDIO_EXTENSIONS = (".mp4", ".m4a", ".opus", ".ogg", ".mp3", ".wav")


def normalize_audio_to_wav(audio_path: Path) -> Path:
    """Convert an audio container to a temporary 16 kHz mono PCM WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to read WhatsApp and uploaded audio. "
            "Install it on Windows with `winget install Gyan.FFmpeg`, "
            "restart the terminal, and verify with `ffmpeg -version`."
        )
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        output.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg took longer than 120 seconds to convert the audio") from error
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        details = completed.stderr.strip().splitlines()[-1:]
        detail = f" Details: {details[0]}" if details else ""
        raise RuntimeError(f"ffmpeg could not decode this audio file.{detail}")
    return output


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: int = CHUNK_SECONDS,
) -> list[tuple[float, np.ndarray]]:
    if sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("sample_rate and chunk_seconds must be positive")

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("audio must be mono")

    window_size = sample_rate * chunk_seconds
    return [
        (start / sample_rate, samples[start : start + window_size])
        for start in range(0, samples.size, window_size)
    ]