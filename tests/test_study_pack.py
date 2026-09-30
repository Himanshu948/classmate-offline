import unittest

from classmate_offline.study_pack import (
    MAX_TRANSCRIPT_WORDS,
    parse_flashcards,
    parse_study_pack,
    split_transcript,
    validate_transcript,
)


class ParseStudyPackTests(unittest.TestCase):
    def test_rejects_transcripts_over_local_context_limit(self) -> None:
        transcript = "word " * (MAX_TRANSCRIPT_WORDS + 1)

        chunks = split_transcript(validate_transcript(transcript))

        self.assertEqual([len(chunk.split()) for chunk in chunks], [1_000, 1])

    def test_empty_transcript_has_no_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "Transcript is empty"):
            validate_transcript("  ")
        self.assertEqual(split_transcript(""), [])

    def test_parses_json_followed_by_model_stop_tokens(self) -> None:
        output = "SUMMARY: Lecture summary.\nQ: What is the topic?\nA: Lecture summary."

        summary, flashcards = parse_study_pack(output)

        self.assertEqual(summary, "Lecture summary.")
        self.assertEqual(len(flashcards), 1)

    def test_drops_unsupported_flashcard_answers(self) -> None:
        output = "SUMMARY: Binary uses two digits.\nQ: How many digits?\nA: Two\nQ: What color?\nA: Blue"

        summary, _ = parse_study_pack(output)
        flashcards, dropped = parse_flashcards(output, "Binary uses two digits.", 2)

        self.assertEqual(summary, "Binary uses two digits.")
        self.assertEqual(len(flashcards), 1)
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()