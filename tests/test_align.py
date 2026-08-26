from pathlib import Path

import pytest

from quest1.audio.align import AlignError, _spell_out_numbers, align_words


def test_align_words_rejects_empty_target_text():
    """Validated before any audio decode -- must not require real media."""
    with pytest.raises(AlignError, match="no words"):
        align_words(Path("does-not-exist.mp4"), "   ", 0.0, 1.0, aligner=None)


def test_align_words_rejects_missing_video():
    with pytest.raises(AlignError):
        align_words(Path("does-not-exist.mp4"), "hello world", 0.0, 1.0, aligner=None)


def test_spell_out_numbers_expands_a_decimal():
    """Real crash observed in practice: the forced-alignment model's
    vocabulary is 27 Latin letters, no digits, so "1.4 billion years" raised
    a bare KeyError('1') from the tokenizer. Numbers must become words
    before alignment ever sees them."""
    assert _spell_out_numbers("1.4 billion years") == "one point four billion years"


def test_spell_out_numbers_expands_a_plain_integer():
    assert _spell_out_numbers("chapter 7") == "chapter seven"


def test_spell_out_numbers_leaves_text_without_digits_untouched():
    assert _spell_out_numbers("no numbers here") == "no numbers here"


def test_spell_out_numbers_output_has_no_digit_characters():
    """The real invariant that matters: whatever comes out must be safe to
    hand to a Latin-letters-only tokenizer."""
    result = _spell_out_numbers("in 1.4 billion years, or maybe 2 billion")
    assert not any(char.isdigit() for char in result)
