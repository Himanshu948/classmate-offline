from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from classmate_offline.audio import SAMPLE_RATE, chunk_audio, normalize_audio_to_wav
from classmate_offline.paths import WHISPER_MODEL_DIR

MODEL_ID = "Systran/faster-whisper-small"
DEFAULT_LANGUAGE = "en"
DEFAULT_TASK = "translate"
DEFAULT_DECODING_OPTIONS = {
    "beam_size": 5,
    "best_of": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "vad_filter": True,
}


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    segments: tuple[TranscriptSegment, ...]
    language: str
    model: str
    device: str
    latency_seconds: float
    quality_warning: str | None

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()


class ASRBackend(Protocol):
    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> TranscriptionResult: ...


class WhisperCpuBackend:
    def __init__(self, model_dir: Path = WHISPER_MODEL_DIR) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Whisper model not found at {model_dir}. "
                "Run `python -m classmate_offline.download_model` while online first."
            )

        self._model = WhisperModel(
            str(model_dir),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = DEFAULT_LANGUAGE,
        task: str = DEFAULT_TASK,
    ) -> TranscriptionResult:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        started = perf_counter()
        normalized_path = normalize_audio_to_wav(audio_path)
        try:
            audio = decode_audio(str(normalized_path), sampling_rate=SAMPLE_RATE)
            if audio.size == 0:
                raise ValueError("Audio file contains no decodable audio")

            transcript_segments: list[TranscriptSegment] = []
            detected_language = language or "unknown"
            for offset, audio_chunk in chunk_audio(audio):
                segments, info = self._model.transcribe(
                    audio_chunk,
                    language=language,
                    task=task,
                    **DEFAULT_DECODING_OPTIONS,
                )
                if detected_language == "unknown":
                    detected_language = info.language
                transcript_segments.extend(
                    TranscriptSegment(
                        start=offset + segment.start,
                        end=offset + segment.end,
                        text=segment.text,
                    )
                    for segment in segments
                )
        finally:
            normalized_path.unlink(missing_ok=True)

        return TranscriptionResult(
            segments=tuple(transcript_segments),
            language=detected_language,
            model=MODEL_ID,
            device="CPU (AMD Ryzen 5 5500U)",
            latency_seconds=perf_counter() - started,
            quality_warning=transcript_quality_warning(
                " ".join(segment.text.strip() for segment in transcript_segments)
            ),
        )


def transcript_quality_warning(text: str) -> str | None:
    words = [word.strip(".,!?;:").lower() for word in text.split()]
    if len(words) < 6:
        return "Transcript is very short; review it against the audio."

    for phrase_size in (2, 3):
        for index in range(len(words) - (phrase_size * 2) + 1):
            phrase = words[index : index + phrase_size]
            if words[index + phrase_size : index + (phrase_size * 2)] == phrase:
                return "Warning: transcript contains repeated phrases; review it against the audio."
    return None