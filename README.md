# ClassMate Offline

Local lecture transcription and study-pack generation for students. The app runs fully on the local CPU, uses Whisper for transcription, and generates a transcript-grounded summary plus flashcards with Phi-3.
## Why
Many students follow lectures in a Hindi-English mix, and not everyone has a
reliable internet connection. ClassMate Offline turns a lecture recording into an
English transcript, summary, and flashcards using models that run entirely on the
laptop, so the audio never leaves the device.

## Tested on

- Device: AMD Ryzen 5 5500U
- Compute: CPU only
- ASR: Whisper Small with translate-to-English
- Real ASR check: about 11 s for a ~45 s audio clip on this machine
- Real Phi-3 run: about 56.28 s to generate a study pack from a short transcript on the same CPU
- NPU profiling: not done; the repo is intended for Snapdragon workflows but this machine was validated only on CPU

## Setup

1. Create the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

2. Install ffmpeg if you want to upload common mobile audio formats such as `.mp4`, `.m4a`, `.opus`, `.ogg`, `.mp3`, and `.wav`:

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
```

3. Download the local models once while online:

```powershell
python -m classmate_offline.download_model
python -m classmate_offline.download_llm
```

4. Start the UI:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m classmate_offline.ui
```

Open http://127.0.0.1:7860 in a browser.

## Architecture

```mermaid
flowchart LR
  A[Upload or record audio] --> B[ffmpeg normalize to 16 kHz mono WAV]
  B --> C[Whisper Small CPU int8\ntranslate-to-English]
  C --> D[Transcript + ASR timing]
  D --> E[Phi-3 Mini Q4_0\nlocal summary + flashcards]
  E --> F[Gradio study pack UI]
```

## Current default behavior

- Whisper language hint defaults to `auto`, but the actual transcription path uses `translate` to English.
- The app accepts `.mp4`, `.m4a`, `.opus`, `.ogg`, `.mp3`, and `.wav` uploads plus microphone recording.
- Flashcards are not forced to a fixed count if the model cannot justify them; unsupported cards are dropped and the pack is marked partial.
- Each generated pack includes a review warning that tells the user to verify the summary and cards against the transcript.

## Limitations

- Hinglish speech is translated to English and can be imperfect.
- Noisy audio can trigger repeated-phrase warnings and weaker transcripts.
- Whisper processes audio in fixed windows, so a word or sentence can be split at a boundary.
- Phi-3 is strongest on English and may produce a partial pack when the transcript does not support more cards.
- NPU profiling was not done in this repo; the intended Snapdragon workflow was not validated on this machine.

## Snapdragon roadmap (planned, NOT implemented or tested)

This version runs on CPU only. The planned next step is to swap the ASR backend
for a Whisper model from Qualcomm AI Hub, run through ONNX Runtime's QNN
Execution Provider on a Snapdragon X-series Windows ARM64 laptop, and then
benchmark NPU against CPU. Per Qualcomm's documentation at the time of writing,
qai-hub-models requires x64 Python (3.10-3.13), while the QNN runtime for local
NPU inference needs ARM64 Python 3.11, so two separate environments are needed.
None of this has been run by the author yet.

## Credits

OpenAI Whisper (MIT), Microsoft Phi-3 Mini (MIT), GPT4All runtime, Gradio, ffmpeg.

## Notes

- The app is verified for local CPU use only.
- The first model download needs internet access; later runs can operate locally from the downloaded model files.
- There is no claim of NPU acceleration in this project.
