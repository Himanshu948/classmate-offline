# Two-Minute Demo Script

1. **0:00-0:15** Say: "ClassMate Offline turns a lecture into revision material locally on the CPU. This app does not claim NPU acceleration."
2. **0:15-0:30** Open the **Upload / Record** tab and select a real lecture clip. Leave the language hint on `auto` or choose `en`, then click **Transcribe locally**. Point out the transcript and measured ASR latency.
3. **0:30-0:50** Say: "The default ASR path uses Whisper Small with translate-to-English, which keeps the result readable for English lecture content while handling multilingual input."
4. **0:50-1:10** Open the **Study Pack** tab and click **Generate study pack locally**. Point out that the loading state is expected because Phi-3 runs on the CPU.
5. **1:10-1:30** Show the English summary and the grounded flashcards. Explain that the app keeps only cards supported by the transcript; unsupported ones are dropped and the pack is marked partial when necessary.
6. **1:30-1:45** Point to the review warning: "Review this AI-generated summary and flashcards against the transcript." This is part of the app’s honest validation step.
7. **1:45-2:00** Turn on airplane mode or disconnect Wi-Fi. Explain: "The models are already on disk, and the app can continue running locally without cloud API calls; the only network step was initial model preparation."