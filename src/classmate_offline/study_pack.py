import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from gpt4all import GPT4All

from classmate_offline.paths import LLM_MODEL_DIR

LLM_MODEL_NAME = "Phi-3-mini-4k-instruct.Q4_0.gguf"
LLM_MODEL_PATH = LLM_MODEL_DIR / LLM_MODEL_NAME
FLASHCARD_COUNT = 10
MAX_TRANSCRIPT_WORDS = 1_000
REVIEW_WARNING = "Review this AI-generated summary and flashcards against the transcript."


@dataclass(frozen=True)
class Flashcard:
    question: str
    answer: str


@dataclass(frozen=True)
class StudyPack:
    summary: str
    flashcards: tuple[Flashcard, ...]
    model: str
    device: str
    latency_seconds: float
    chunk_summary_seconds: float
    final_pack_seconds: float
    warning: str
    dropped_cards: int
    partial: bool


def validate_transcript(transcript: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty")
    return transcript


def split_transcript(transcript: str, max_words: int = MAX_TRANSCRIPT_WORDS) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    words = transcript.split()
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    ]


def target_flashcard_count(transcript: str) -> int:
    return max(3, min(5, len(transcript.split()) // 40))


def parse_summary(output: str) -> str:
    for line in output.splitlines():
        if line.strip().upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
            if summary:
                return summary
    raise ValueError("The local model returned no SUMMARY line")


def _answer_supported(answer: str, transcript: str) -> bool:
    def normalize(word: str) -> str:
        return word[:-1] if len(word) > 4 and word.endswith("s") else word

    transcript_words = {
        normalize(word) for word in re.findall(r"[a-z0-9]+", transcript.lower())
    }
    stop_words = {"the", "and", "are", "from", "this", "that", "with", "only", "into"}
    answer_words = [
        word
        for word in re.findall(r"[a-z0-9]+", answer.lower())
        if len(word) > 2 or word.isdigit()
    ]
    content_words = [normalize(word) for word in answer_words if word not in stop_words]
    return bool(content_words) and all(word in transcript_words for word in content_words)


def parse_flashcards(
    output: str, transcript: str, expected_count: int
) -> tuple[tuple[Flashcard, ...], int]:
    cards: list[Flashcard] = []
    question: str | None = None
    dropped = 0
    for raw_line in output.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if line.upper().startswith("Q:"):
            if question is not None:
                dropped += 1
            question = line.split(":", 1)[1].strip()
        elif line.upper().startswith("A:") and question:
            answer = line.split(":", 1)[1].strip()
            if answer and _answer_supported(answer, transcript):
                cards.append(Flashcard(question, answer))
            else:
                dropped += 1
            question = None
    if question is not None:
        dropped += 1
    dropped += max(0, expected_count - len(cards) - dropped)
    return tuple(cards[:expected_count]), dropped


def parse_study_pack(output: str) -> tuple[str, tuple[Flashcard, ...]]:
    summary = parse_summary(output)
    cards, _ = parse_flashcards(output, output, FLASHCARD_COUNT)
    return summary, cards


class LocalStudyPackGenerator:
    def __init__(self, model_path: Path = LLM_MODEL_PATH, n_threads: int = 6) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Local LLM not found at {model_path}. "
                "Run `python -m classmate_offline.download_llm` while online first."
            )

        self._model = GPT4All(
            model_path.name,
            model_path=str(model_path.parent),
            allow_download=False,
            device="cpu",
            n_ctx=4096,
            n_threads=n_threads,
        )

    def generate(self, transcript: str) -> StudyPack:
        transcript = validate_transcript(transcript)
        chunks = split_transcript(transcript)
        total_started = perf_counter()
        chunk_summary_seconds = 0.0

        if len(chunks) == 1:
            final_source = chunks[0]
        else:
            summaries: list[str] = []
            for chunk in chunks:
                started = perf_counter()
                summaries.append(self._summarize_chunk(chunk))
                chunk_summary_seconds += perf_counter() - started

            while len(" ".join(summaries).split()) > MAX_TRANSCRIPT_WORDS:
                previous_word_count = len(" ".join(summaries).split())
                summary_groups = split_transcript(" ".join(summaries))
                reduced: list[str] = []
                for summary_group in summary_groups:
                    started = perf_counter()
                    reduced.append(self._summarize_chunk(summary_group))
                    chunk_summary_seconds += perf_counter() - started
                if len(" ".join(reduced).split()) >= previous_word_count:
                    raise ValueError("The local model could not reduce the long transcript summaries")
                summaries = reduced
            final_source = "\n".join(summaries)

        summary_started = perf_counter()
        summary_output = self._generate(
            "Return exactly one line beginning with SUMMARY:. Write it in English using only facts "
            "from this transcript. Do not add outside facts.\n\nTranscript:\n" + final_source,
            max_tokens=180,
        )
        summary = parse_summary(summary_output)
        summary_seconds = perf_counter() - summary_started

        expected_count = target_flashcard_count(final_source)
        cards_started = perf_counter()
        cards_output = self._generate(
            f"Write exactly {expected_count} English flashcards using only this transcript. "
            "Use one Q: line followed by one A: line for each card. Keep answers short and supported. "
            "Do not add commentary or JSON.\n\nTranscript:\n" + final_source,
            max_tokens=expected_count * 80,
        )
        flashcards, dropped_cards = parse_flashcards(cards_output, final_source, expected_count)
        final_pack_seconds = summary_seconds + (perf_counter() - cards_started)
        partial = len(flashcards) < expected_count
        warning = REVIEW_WARNING
        if dropped_cards:
            warning += f" Dropped {dropped_cards} unsupported or malformed flashcard(s)."
        if partial:
            warning += " Partial pack: the real summary is available, but flashcards failed validation."
        return StudyPack(
            summary=summary,
            flashcards=flashcards,
            model=LLM_MODEL_NAME,
            device="CPU (AMD Ryzen 5 5500U)",
            latency_seconds=perf_counter() - total_started,
            chunk_summary_seconds=chunk_summary_seconds,
            final_pack_seconds=final_pack_seconds,
            warning=warning,
            dropped_cards=dropped_cards,
            partial=partial,
        )

    def _summarize_chunk(self, transcript_chunk: str) -> str:
        prompt = (
            "Summarize this lecture transcript chunk in English, even if it is in Hindi or Hinglish. "
            "Use only facts in the chunk, preserve its key facts, and be concise. Return only the summary.\n\n"
            f"Transcript chunk:\n{transcript_chunk}"
        )
        summary = self._generate(prompt, max_tokens=300).strip()
        if not summary:
            raise ValueError("The local model returned an empty chunk summary")
        return summary

    def _generate(self, prompt: str, max_tokens: int) -> str:
        with self._model.chat_session():
            return self._model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=0.1,
                top_k=1,
            )

    def close(self) -> None:
        self._model.close()