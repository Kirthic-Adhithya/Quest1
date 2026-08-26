"""Shared download+transcribe entry points, and the `Result` type the CLI and
web app both assemble once matching, alignment, and frame extraction finish.

`cli.py` and `web/jobs.py` drive matching/alignment/extraction themselves
rather than through a shared function here, since each needs its own
progress reporting -- a concern that doesn't belong in a library function.
`run_transcription`/`transcribe_media` have no such per-caller concern, so
they're shared as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio.extract import extract_audio
from .audio.transcribe import DEFAULT_MODEL_DIR, DEFAULT_MODEL_SIZE, Transcript, load_model, transcribe
from .ingest.downloader import DEFAULT_QUALITY, Media, fetch, probe
from .inputs import Job
from .search.matcher import Candidate
from .video.frames import FrameHit


def transcribe_media(
    media: Media,
    media_dir: Path = Path("data/media"),
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dir: Path = DEFAULT_MODEL_DIR,
    language: str | None = None,
) -> Transcript:
    """Transcribe `media`'s audio (if not already transcribed for this
    language), caching the result to disk keyed on the media's filename and
    language. Split out from `run_transcription` so a caller that already
    has a local file (an uploaded video, not one this pipeline downloaded)
    can reuse the same transcription + caching logic without going through
    `fetch()`.
    """
    lang_tag = language or "auto"
    transcript_path = media_dir / f"{media.path.stem}.{lang_tag}.transcript.json"
    if transcript_path.exists():
        return Transcript.from_json(transcript_path.read_text(encoding="utf-8"))

    audio_path = media_dir / f"{media.path.stem}.wav"
    audio = extract_audio(media.path, audio_path)

    model = load_model(model_size, download_root=model_dir)
    transcript = transcribe(audio, model, language=language)

    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    return transcript


def run_transcription(
    job: Job,
    media_dir: Path = Path("data/media"),
    quality: str = DEFAULT_QUALITY,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dir: Path = DEFAULT_MODEL_DIR,
    language: str | None = None,
) -> tuple[Media, Transcript]:
    """Download the media (if not already cached), then transcribe its audio.
    Both steps are cached to disk, so a re-run only redoes whatever actually
    changed.

    `language=None` auto-detects from the first ~30s of audio, which can
    mis-detect on a non-speech opening; pass the language explicitly when
    it's known.
    """
    download = fetch(job, media_dir, quality)
    media = probe(download.path, download.title)
    transcript = transcribe_media(media, media_dir, model_size, model_dir, language)
    return media, transcript


@dataclass(frozen=True)
class Result:
    """The final answer: which candidate matched, the refined onset, and the
    actually-decoded frame."""

    media: Media
    match: Candidate
    onset: float  # forced-alignment onset, seconds -- a target, not the answer
    hit: FrameHit  # the decoded frame; hit.index is the reported answer

    @property
    def timestamp(self) -> str:
        """Timestamp of the actually-decoded frame, not the raw alignment
        onset -- the two usually agree to a few ms, but the decoded frame is
        ground truth when they don't."""
        mins, secs = divmod(self.hit.pts_time, 60)
        return f"{int(mins):02d}:{secs:06.3f}"

    @property
    def frame(self) -> int:
        """The reported frame number."""
        return self.hit.index
