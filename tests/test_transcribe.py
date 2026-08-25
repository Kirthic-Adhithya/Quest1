from quest1.audio.transcribe import Transcript, Word


def test_transcript_json_round_trip():
    original = Transcript(
        words=[
            Word(text="my", start=325.26, end=325.40, prob=0.98),
            Word(text="mind", start=325.46, end=325.70, prob=0.97),
        ],
        language="en",
        language_prob=1.0,
    )
    restored = Transcript.from_json(original.to_json())
    assert restored == original
