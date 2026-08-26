import json
from fractions import Fraction
from pathlib import Path

import pytest

from quest1.ingest.downloader import (
    CACHE_INDEX,
    IngestError,
    Media,
    _evict_other_videos,
    _evict_stale,
    lookup_cached,
    probe,
)


def make_media(**overrides) -> Media:
    defaults = dict(
        path=Path("data/media/example.mp4"),
        title="Example",
        fps=Fraction(25, 1),
        frame_count=81525,
        duration=3261.0,
        width=640,
        height=480,
    )
    return Media(**{**defaults, **overrides})


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(IngestError, match="No such media file"):
        probe(tmp_path / "nope.mp4")


def test_probe_rejects_non_media_file(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(IngestError):
        probe(junk)


def test_lookup_cached_returns_none_without_index(tmp_path):
    assert lookup_cached(tmp_path, "https://example.com/v/1", "sd") is None


def test_lookup_cached_round_trip(tmp_path):
    video = tmp_path / "abc.mp4"
    video.write_bytes(b"x" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {"https://example.com/v/1": {"filename": "abc.mp4", "title": "T", "quality": "sd"}}
        ),
        encoding="utf-8",
    )
    hit = lookup_cached(tmp_path, "https://example.com/v/1", "sd")
    assert hit is not None
    assert hit.path == video and hit.title == "T" and hit.from_cache


def test_lookup_cached_misses_on_quality_mismatch(tmp_path):
    """A file fetched at 'sd' must never satisfy a request for 'hd'."""
    (tmp_path / "abc.mp4").write_bytes(b"x" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {"https://example.com/v/1": {"filename": "abc.mp4", "title": "T", "quality": "sd"}}
        ),
        encoding="utf-8",
    )
    assert lookup_cached(tmp_path, "https://example.com/v/1", "hd") is None


def test_lookup_cached_misses_on_legacy_entry_without_quality(tmp_path):
    """An index entry written before quality-keying existed must not be trusted."""
    (tmp_path / "abc.mp4").write_bytes(b"x" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps({"https://example.com/v/1": {"filename": "abc.mp4", "title": "T"}}),
        encoding="utf-8",
    )
    assert lookup_cached(tmp_path, "https://example.com/v/1", "sd") is None


def test_lookup_cached_ignores_missing_and_empty_files(tmp_path):
    (tmp_path / "empty.mp4").write_bytes(b"")
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {
                "https://example.com/gone": {
                    "filename": "gone.mp4", "title": "G", "quality": "sd"
                },
                "https://example.com/empty": {
                    "filename": "empty.mp4", "title": "E", "quality": "sd"
                },
            }
        ),
        encoding="utf-8",
    )
    assert lookup_cached(tmp_path, "https://example.com/gone", "sd") is None
    assert lookup_cached(tmp_path, "https://example.com/empty", "sd") is None


def test_corrupt_index_is_ignored_not_fatal(tmp_path):
    (tmp_path / CACHE_INDEX).write_text("{not json", encoding="utf-8")
    assert lookup_cached(tmp_path, "https://example.com/v/1", "sd") is None


def test_evict_stale_removes_old_quality_file(tmp_path):
    """Fetching a URL at a new quality must delete the previously cached file,
    not leave it orphaned on disk alongside the new one."""
    old = tmp_path / "abc.mp4"
    old.write_bytes(b"x" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {"https://example.com/v/1": {"filename": "abc.mp4", "title": "T", "quality": "sd"}}
        ),
        encoding="utf-8",
    )
    _evict_stale(tmp_path, "https://example.com/v/1")
    assert not old.exists()


def test_evict_stale_removes_partial_download_leftovers(tmp_path):
    """A previous run's .part/.ytdl state must not survive into the next
    attempt -- resuming into a differently-selected format is what produced an
    unfixed raw MPEG-TS file wearing an .mp4 extension in practice."""
    (tmp_path / "abc.mp4").write_bytes(b"x" * 10)
    (tmp_path / "abc.mp4.part-Frag22.part").write_bytes(b"y")
    (tmp_path / "abc.mp4.ytdl").write_text("{}", encoding="utf-8")
    index_path = tmp_path / CACHE_INDEX
    index_path.write_text(
        json.dumps(
            {"https://example.com/v/1": {"filename": "abc.mp4", "title": "T", "quality": "sd"}}
        ),
        encoding="utf-8",
    )
    _evict_stale(tmp_path, "https://example.com/v/1")
    assert not (tmp_path / "abc.mp4").exists()
    assert not (tmp_path / "abc.mp4.part-Frag22.part").exists()
    assert not (tmp_path / "abc.mp4.ytdl").exists()
    assert index_path.exists()  # the cache index itself must never be swept up


def test_evict_stale_is_a_noop_for_unknown_url(tmp_path):
    _evict_stale(tmp_path, "https://example.com/never-seen")  # must not raise


def test_evict_other_videos_removes_everything_but_the_kept_url(tmp_path):
    """Deploying should never accumulate every video ever fetched -- switching
    to a new URL must free the previous video's files, not keep both."""
    (tmp_path / "old.mp4").write_bytes(b"x" * 10)
    (tmp_path / "old.wav").write_bytes(b"y" * 10)
    (tmp_path / "old.en.transcript.json").write_text("{}", encoding="utf-8")
    (tmp_path / "keep.mp4").write_bytes(b"z" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {
                "https://example.com/old": {"filename": "old.mp4", "title": "Old", "quality": "sd"},
                "https://example.com/keep": {"filename": "keep.mp4", "title": "Keep", "quality": "sd"},
            }
        ),
        encoding="utf-8",
    )

    _evict_other_videos(tmp_path, "https://example.com/keep")

    assert not (tmp_path / "old.mp4").exists()
    assert not (tmp_path / "old.wav").exists()
    assert not (tmp_path / "old.en.transcript.json").exists()
    assert (tmp_path / "keep.mp4").exists()

    remaining = json.loads((tmp_path / CACHE_INDEX).read_text(encoding="utf-8"))
    assert list(remaining.keys()) == ["https://example.com/keep"]


def test_evict_other_videos_is_a_noop_when_only_the_kept_url_is_cached(tmp_path):
    (tmp_path / "keep.mp4").write_bytes(b"x" * 10)
    (tmp_path / CACHE_INDEX).write_text(
        json.dumps(
            {"https://example.com/keep": {"filename": "keep.mp4", "title": "Keep", "quality": "sd"}}
        ),
        encoding="utf-8",
    )
    _evict_other_videos(tmp_path, "https://example.com/keep")
    assert (tmp_path / "keep.mp4").exists()


def test_evict_other_videos_is_a_noop_without_an_index(tmp_path):
    _evict_other_videos(tmp_path, "https://example.com/keep")  # must not raise
    assert not (tmp_path / CACHE_INDEX).exists()  # nothing written for an empty index
