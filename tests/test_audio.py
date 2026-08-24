from pathlib import Path

import pytest

from quest1.audio.extract import AudioExtractError, extract_audio


def test_extract_audio_rejects_missing_file(tmp_path):
    with pytest.raises(AudioExtractError):
        extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav")


def test_extract_audio_rejects_non_media_file(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(AudioExtractError):
        extract_audio(junk, tmp_path / "out.wav")


def test_extract_audio_reuses_existing_output(tmp_path):
    """A non-empty destination is treated as cached and never touches the
    (missing, invalid) source -- decoding a long episode is expensive enough
    to be worth skipping on a re-run, same as the download cache."""
    dest = tmp_path / "out.wav"
    dest.write_bytes(b"already decoded")
    result = extract_audio(tmp_path / "does-not-exist.mp4", dest)
    assert result == dest
    assert dest.read_bytes() == b"already decoded"
