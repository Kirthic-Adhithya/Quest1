from pathlib import Path

import pytest

from quest1.video.frames import FrameExtractError, extract_frame
from fractions import Fraction


def test_extract_frame_rejects_negative_onset():
    with pytest.raises(FrameExtractError, match=">= 0"):
        extract_frame(Path("irrelevant.mp4"), -1.0, Fraction(25, 1))


def test_extract_frame_rejects_missing_file():
    with pytest.raises(FrameExtractError):
        extract_frame(Path("does-not-exist.mp4"), 1.0, Fraction(25, 1))
