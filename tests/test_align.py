from pathlib import Path

import pytest

from quest1.audio.align import AlignError, align_words


def test_align_words_rejects_empty_target_text():
    """Validated before any audio decode -- must not require real media."""
    with pytest.raises(AlignError, match="no words"):
        align_words(Path("does-not-exist.mp4"), "   ", 0.0, 1.0, aligner=None)


def test_align_words_rejects_missing_video():
    with pytest.raises(AlignError):
        align_words(Path("does-not-exist.mp4"), "hello world", 0.0, 1.0, aligner=None)
