from pathlib import Path

import gradio as gr

from classmate_offline.asr import WhisperCpuBackend
from classmate_offline.study_pack import LocalStudyPackGenerator, StudyPack


_asr_backend: WhisperCpuBackend | None = None
_study_pack_generator: LocalStudyPackGenerator | None = None


def _get_asr_backend() -> WhisperCpuBackend:
    global _asr_backend
    if _asr_backend is None:
        _asr_backend = WhisperCpuBackend()
    return _asr_backend


def _get_study_pack_generator() -> LocalStudyPackGenerator:
    global _study_pack_generator
    if _study_pack_generator is None:
        _study_pack_generator = LocalStudyPackGenerator()
    return _study_pack_generator


def transcribe_audio(
    uploaded_path: str | None,
    recorded_path: str | None,
    language: str,
) -> tuple[str, str, str, str]:
    audio_path = uploaded_path or recorded_path
    if not audio_path:
        raise gr.Error("Choose an audio file or record audio first.")

    try:
        result = _get_asr_backend().transcribe(
            Path(audio_path),
            language=None if language == "auto" else language,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    timing = (
        f"ASR latency: {result.latency_seconds:.2f} s | "
        f"detected language: {result.language} | task: translate-to-English | "
        f"device: {result.device}"
    )
    status = result.quality_warning or "Transcript ready. Open Study Pack to continue."
    return result.text, timing, result.text, status


def _render_flashcards(study_pack: StudyPack) -> str:
    return "\n\n".join(
        f"**{index}. {card.question}**\n\n{card.answer}"
        for index, card in enumerate(study_pack.flashcards, start=1)
    )


def generate_study_pack(transcript: str | None) -> tuple[str, str, str, str]:
    if not transcript or not transcript.strip():
        raise gr.Error("Transcribe an audio recording before generating a study pack.")

    generator = _get_study_pack_generator()
    try:
        study_pack = generator.generate(transcript)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    timing = (
        f"Study-pack total: {study_pack.latency_seconds:.2f} s | "
        f"chunk summaries: {study_pack.chunk_summary_seconds:.2f} s | "
        f"final pack: {study_pack.final_pack_seconds:.2f} s | "
        f"cards shown: {len(study_pack.flashcards)} | "
        f"cards dropped: {study_pack.dropped_cards} | device: {study_pack.device}"
    )
    return study_pack.summary, _render_flashcards(study_pack), timing, study_pack.warning


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ClassMate Offline") as app:
        gr.Markdown("# ClassMate Offline\nLocal lecture transcription and study pack generation")
        transcript_state = gr.State("")

        with gr.Tab("Upload / Record"):
            upload_input = gr.File(
                file_types=[".mp4", ".m4a", ".opus", ".ogg", ".mp3", ".wav"],
                type="filepath",
                label="Upload lecture audio",
            )
            microphone_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Or record with microphone",
            )
            language = gr.Dropdown(
                choices=["auto", "hi", "en"],
                value="auto",
                label="Whisper language hint",
            )
            transcribe_button = gr.Button("Transcribe locally", variant="primary")
            transcript_output = gr.Textbox(label="Transcript", lines=12, interactive=False)
            asr_timing = gr.Markdown()
            asr_status = gr.Markdown()

        with gr.Tab("Study Pack"):
            gr.Markdown("Generate a study pack from the transcript produced in the first tab.")
            generate_button = gr.Button("Generate study pack locally", variant="primary")
            summary_output = gr.Markdown()
            flashcards_output = gr.Markdown()
            llm_timing = gr.Markdown()
            review_warning = gr.Markdown()

        transcribe_button.click(
            transcribe_audio,
            inputs=[upload_input, microphone_input, language],
            outputs=[transcript_output, asr_timing, transcript_state, asr_status],
            api_name="transcribe",
        )
        generate_button.click(
            generate_study_pack,
            inputs=transcript_state,
            outputs=[summary_output, flashcards_output, llm_timing, review_warning],
            api_name="study_pack",
        )

    return app


if __name__ == "__main__":
    build_app().queue().launch()