import argparse
from pathlib import Path

from classmate_offline.asr import WhisperCpuBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe lecture audio locally.")
    parser.add_argument("audio", type=Path, help="Audio file to transcribe")
    parser.add_argument("--language", choices=("auto", "hi", "en"), default="en")
    args = parser.parse_args()

    result = WhisperCpuBackend().transcribe(
        args.audio,
        language=None if args.language == "auto" else args.language,
    )
    print(
        f"device={result.device} model={result.model} "
        f"language={result.language} task=translate "
        f"latency_seconds={result.latency_seconds:.2f}"
    )
    if result.quality_warning:
        print(f"warning={result.quality_warning}")
    for segment in result.segments:
        print(f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.text.strip()}")


if __name__ == "__main__":
    main()