from quest1.audio.transcribe import Transcript, Word


def test_joined_text_space_separates_words_in_order():
    t = Transcript(
        words=[
            Word(text="my", start=326.18, end=326.31, prob=0.99),
            Word(text="mind", start=326.31, end=326.58, prob=0.98),
            Word(text="rebels", start=326.58, end=327.02, prob=0.95),
        ],
        language="en",
        language_prob=0.99,
    )
    assert t.joined_text() == "my mind rebels"


def test_joined_text_empty_transcript():
    assert Transcript(words=[], language="en", language_prob=1.0).joined_text() == ""
