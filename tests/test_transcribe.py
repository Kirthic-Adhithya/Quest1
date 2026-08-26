from quest1.audio.transcribe import (
    DEFAULT_MODEL_SIZE,
    DISTIL_MODEL_SIZE,
    LANGUAGES,
    Transcript,
    Word,
    pick_model_size,
)


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


def test_pick_model_size_uses_distil_only_for_explicit_english():
    assert pick_model_size("en") == DISTIL_MODEL_SIZE


def test_pick_model_size_uses_large_for_auto_detect():
    """Auto-detect must never route to the English-only model, even though
    detection could land on English -- a misdetected non-English video would
    then hit a model that can't represent it at all."""
    assert pick_model_size(None) == DEFAULT_MODEL_SIZE


def test_pick_model_size_uses_large_for_other_languages():
    assert pick_model_size("fr") == DEFAULT_MODEL_SIZE
    assert pick_model_size("hi") == DEFAULT_MODEL_SIZE


def test_languages_table_has_no_empty_names():
    assert all(name.strip() for name in LANGUAGES.values())
    assert "en" in LANGUAGES
