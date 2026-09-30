import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from classmate_offline.audio import chunk_audio, normalize_audio_to_wav
from classmate_offline.asr import transcript_quality_warning


class ChunkAudioTests(unittest.TestCase):
    def test_splits_into_30_second_windows_and_offsets_remainder(self) -> None:
        audio = np.zeros(31 * 16_000, dtype=np.float32)

        chunks = chunk_audio(audio)

        self.assertEqual([offset for offset, _ in chunks], [0.0, 30.0])
        self.assertEqual([len(chunk) for _, chunk in chunks], [480_000, 16_000])

    def test_empty_audio_has_no_chunks(self) -> None:
        self.assertEqual(chunk_audio(np.array([], dtype=np.float32)), [])

    def test_rejects_non_mono_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "mono"):
            chunk_audio(np.zeros((16_000, 2), dtype=np.float32))

    def test_reports_install_steps_when_ffmpeg_is_missing(self) -> None:
        with patch("classmate_offline.audio.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "winget install Gyan.FFmpeg"):
                normalize_audio_to_wav(Path("missing.mp4"))

    def test_warns_on_repeated_phrases(self) -> None:
        warning = transcript_quality_warning("use two use two use two use two")

        self.assertIn("repeated", warning or "")


if __name__ == "__main__":
    unittest.main()