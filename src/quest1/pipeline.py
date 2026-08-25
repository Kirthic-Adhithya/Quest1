"""Shared stage-1/2 entry point, and the `Result` the CLI assembles stages 3-5 into.

`cli.py` drives stages 3-5 (match / align / extract) itself rather than through
functions here, since it needs to print progress and handle the "not found"
near-miss diagnostic between each stage -- concerns that don't belong in a
library function. `run_transcription` (stages 1-2) has no such per-stage
concern, so it is shared as-is.
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


def run_transcription(
    job: Job,
    media_dir: Path = Path("data/media"),
    quality: str = DEFAULT_QUALITY,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dir: Path = DEFAULT_MODEL_DIR,
    language: str | None = None,
) -> tuple[Media, Transcript]:
    """Stages 1-2: get the media locally, then transcribe its audio.

    Each stage's own cache (download cache, decoded-audio cache) means a
    re-run of this function during development only re-does whatever changed.

    `language=None` auto-detects from the first ~30s of audio, which can be
    unreliable when that window is non-speech (observed on the reference video:
    detection landed on "la" at 46% confidence, and the transcript opened with
    a phrase repeated three times verbatim -- a known Whisper failure mode
    triggered by a wrong language tag). Pass the known language explicitly
    when it's known ahead of time to avoid this class of error.

    The transcript itself is cached to disk (unlike audio/model, this wasn't
    obvious until a re-run with no cache silently re-ran a ~10-minute GPU
    transcription for the reference video). Caching is keyed on `language`,
    since re-transcribing with a different language argument must not return
    a transcript produced under the previous one.
    """
    download = fetch(job, media_dir, quality)
    media = probe(download.path, download.title)

    lang_tag = language or "auto"
    transcript_path = media_dir / f"{media.path.stem}.{lang_tag}.transcript.json"
    if transcript_path.exists():
        return media, Transcript.from_json(transcript_path.read_text(encoding="utf-8"))

    audio_path = media_dir / f"{media.path.stem}.wav"
    audio = extract_audio(media.path, audio_path)

    model = load_model(model_size, download_root=model_dir)
    transcript = transcribe(audio, model, language=language)

    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    return media, transcript


@dataclass(frozen=True)
class Result:
    media: Media
    match: Candidate
    onset: float  # forced-alignment-refined onset, seconds (a target, not the answer)
    hit: FrameHit  # the actually-decoded frame; hit.index is the reported answer

    @property
    def timestamp(self) -> str:
        """Timestamp of the actually-decoded frame (`hit.pts_time`), not the
        raw alignment onset -- the two usually agree to a few ms, but when
        they don't, the decoded frame is ground truth (see `video/frames.py`
        and the frame_at() floor-vs-round bug this distinction caught)."""
        mins, secs = divmod(self.hit.pts_time, 60)
        return f"{int(mins):02d}:{secs:06.3f}"

    @property
    def frame(self) -> int:
        return self.hit.index
