"""Download media from a URL via yt-dlp, then probe its properties with PyAV.

`fetch()` and `probe()` are kept separate so either can be used alone. Frame
rate is a `Fraction` (e.g. 25/1, 30000/1001), never a float, since everything
downstream converts between time and frame numbers using it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import av
import imageio_ffmpeg
import yt_dlp

from ..inputs import Job

#: ok.ru exposes named rungs rather than numeric heights; yt-dlp maps these
#: for other sites too, falling back to "best available at or below this".
QUALITY_CHOICES = ("mobile", "lowest", "low", "sd", "hd", "best")
DEFAULT_QUALITY = "sd"

#: Height-bounded format selectors, ranked by real per-format resolution
#: metadata rather than a site's own tier names (ok.ru's named tiers carry no
#: resolution/bitrate at all, so name-based preference is unranked data
#: racing ranked data). Each tries a split video+audio pair first (needed on
#: sites like YouTube that serve no single muxed format), falling back to a
#: single muxed format where one exists.
_GENERIC_FALLBACK = {
    "mobile": "worstvideo[height<=240]+worstaudio/worst[height<=240]/worst",
    "lowest": "worstvideo+worstaudio/worst",
    "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "sd": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "hd": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    "best": "bestvideo+bestaudio/best",
}


def _format_selector(quality: str) -> str:
    """yt-dlp format-selector string for a quality name."""
    return _GENERIC_FALLBACK[quality]


#: ok.ru intermittently resets connections mid-handshake; retried rather than
#: assumed to succeed first try.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 2.0

#: ok.ru's extractor tries a rich "desktop" page (full format ladder with real
#: resolution) and silently falls back to a crippled "mobile" page (no
#: resolution data) on any error -- which succeeds without raising, so the
#: ordinary retry loop never sees it as a failure. Retried separately until
#: the rich ladder is actually observed.
RICH_PROBE_ATTEMPTS = 6
RICH_PROBE_BACKOFF = 1.5

#: Maps URL -> downloaded filename, so a repeat run never touches the network.
CACHE_INDEX = ".cache.json"

CR = "\r"


def _cache_key(url: str) -> str:
    """Normalize a URL for cache-key comparison, so trivial variations of
    the same resource don't miss an otherwise-identical cache entry and
    silently trigger a redownload. Confirmed as a real cause of one: the
    same video, requested once as http:// and once as https://, was cached
    under two different keys and never hit on the second request. Scheme is
    folded to https (http/https serve the same resource on every site this
    targets), host is lower-cased, and a trailing slash on the path is
    dropped; the query string is kept since it can carry a real video id.
    """
    parts = urlsplit(url)
    scheme = "https" if parts.scheme in ("http", "https") else parts.scheme.lower()
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((scheme, parts.netloc.lower(), path, parts.query, ""))


class IngestError(RuntimeError):
    """Raised when the media cannot be fetched or read."""


@dataclass(frozen=True)
class Download:
    """A local media file plus the title reported by the extractor."""

    path: Path
    title: str
    from_cache: bool = False


@dataclass(frozen=True)
class Media:
    """A downloaded video and the properties needed to locate a frame in it."""

    path: Path
    title: str
    fps: Fraction
    frame_count: int
    duration: float
    width: int
    height: int

    @property
    def size_bytes(self) -> int:
        """File size on disk, in bytes."""
        return self.path.stat().st_size

    def describe(self) -> str:
        """Human-readable summary: title, size, resolution, fps, duration."""
        mins, secs = divmod(self.duration, 60)
        return (
            f"Title      : {self.title}\n"
            f"File       : {self.path}\n"
            f"Size       : {self.size_bytes / 1_048_576:.1f} MiB\n"
            f"Resolution : {self.width}x{self.height}\n"
            f"Frame rate : {self.fps} ({float(self.fps):.3f} fps)\n"
            f"Duration   : {int(mins):02d}:{secs:05.2f} ({self.duration:.2f} s)\n"
            f"Frames     : {self.frame_count}"
        )


def _read_index(dest_dir: Path) -> dict:
    """Load the cache index, or an empty dict if it's missing/corrupt.

    Keys are re-normalised on every load, not just on write -- so an index
    entry written before cache-key normalisation existed (or written by an
    older version of this function) still matches on the next lookup,
    instead of permanently missing and redownloading.
    """
    try:
        raw = json.loads((dest_dir / CACHE_INDEX).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {_cache_key(key): entry for key, entry in raw.items()}


def _write_index(dest_dir: Path, index: dict) -> None:
    """Save the cache index; a write failure is logged, never fatal."""
    try:
        (dest_dir / CACHE_INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write cache index: {exc}", file=sys.stderr)


def lookup_cached(dest_dir: Path, url: str, quality: str) -> Download | None:
    """Previously downloaded file for `url` at `quality`, or None.

    Quality is part of the cache key: a file fetched at "sd" is never handed
    back for an "hd" request.
    """
    entry = _read_index(dest_dir).get(_cache_key(url))
    if not entry or entry.get("quality") != quality:
        return None
    path = dest_dir / entry["filename"]
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return Download(path=path, title=entry.get("title", path.stem), from_cache=True)


def _evict_stale(dest_dir: Path, url: str) -> None:
    """Delete `url`'s previous file plus any partial-download leftovers
    (`.part`, `.part-FragN.part`, `.ytdl`) sharing its filename stem, so a
    fresh attempt never resumes into state left by a different selection."""
    entry = _read_index(dest_dir).get(_cache_key(url))
    if not entry:
        return
    stale = dest_dir / entry["filename"]
    for leftover in dest_dir.glob(f"{stale.stem}.*"):
        if leftover != dest_dir / CACHE_INDEX:
            leftover.unlink()


def _evict_other_videos(dest_dir: Path, keep_url: str) -> None:
    """Delete every cached video except `keep_url` and its files, so the
    cache holds at most one video at a time -- a deployed instance processing
    one video per run should not accumulate every video ever fetched."""
    keep_key = _cache_key(keep_url)
    index = _read_index(dest_dir)
    changed = False
    for key, entry in list(index.items()):
        if key == keep_key:
            continue
        stale = dest_dir / entry["filename"]
        for leftover in dest_dir.glob(f"{stale.stem}.*"):
            if leftover != dest_dir / CACHE_INDEX:
                leftover.unlink(missing_ok=True)
        del index[key]
        changed = True
    if changed:
        _write_index(dest_dir, index)


def _has_resolution_ladder(info: dict) -> bool:
    """True once extracted metadata carries real per-format resolution
    (ok.ru's crippled fallback formats never report height at all)."""
    return any(f.get("height") for f in info.get("formats", []))


def _extract_rich(url: str, options: dict) -> dict:
    """Extract metadata, retrying until the rich resolution ladder appears.

    Kept separate from `fetch`'s own retry loop: the crippled fallback is not
    an exception, just worse data, so it never triggers an error-based retry.
    """
    probe_options = {**options, "progress_hooks": []}
    best_seen: dict | None = None

    for attempt in range(1, RICH_PROBE_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(probe_options) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError:
            time.sleep(RICH_PROBE_BACKOFF * attempt)
            continue

        if _has_resolution_ladder(info):
            return info
        if best_seen is None or len(info.get("formats", [])) > len(best_seen.get("formats", [])):
            best_seen = info
        if attempt < RICH_PROBE_ATTEMPTS:
            time.sleep(RICH_PROBE_BACKOFF * attempt)

    if best_seen is None:
        raise IngestError(f"Could not read any format metadata for {url}")

    print(
        "warning: ok.ru only offered its limited mobile format list after "
        f"{RICH_PROBE_ATTEMPTS} attempts; quality selection may not reflect "
        "the true best available.",
        file=sys.stderr,
    )
    return best_seen


def _progress(status: dict) -> None:
    """Single-line download progress, overwritten in place."""
    if status["status"] == "downloading":
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes", 0)
        if total:
            pct = 100 * done / total
            line = f"  {pct:5.1f}%  {done / 1_048_576:7.1f} / {total / 1_048_576:.1f} MiB"
            print(CR + line, end="", file=sys.stderr, flush=True)
    elif status["status"] == "finished":
        print(CR + "  download complete" + " " * 30, file=sys.stderr)


def fetch(
    job: Job,
    dest_dir: Path,
    quality: str = DEFAULT_QUALITY,
    show_progress: bool = True,
) -> Download:
    """Download the media for `job` into `dest_dir`, reusing a cached copy of
    the same (url, quality) pair if one exists. Evicts every other cached
    video first (single-video cache) and retries connection resets with a
    linear backoff."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _evict_other_videos(dest_dir, job.url)

    cached = lookup_cached(dest_dir, job.url, quality)
    if cached is not None:
        return cached

    _evict_stale(dest_dir, job.url)

    options = {
        "format": _format_selector(quality),
        # yt-dlp does not discover the ffmpeg bundled by imageio-ffmpeg on its own.
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "merge_output_format": "mp4",
        # A single HLS format downloads as raw MPEG-TS wearing an ".mp4"
        # extension unless explicitly remuxed; force a real container.
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        # The host drops connections mid-transfer; chunking + resume means a
        # drop costs one chunk, not the whole file.
        "continuedl": True,
        "http_chunk_size": 10 * 1024 * 1024,
        # A real-time antivirus scan can briefly lock a just-written HLS
        # fragment file, racing yt-dlp's rename. file_access_retries is a
        # separate knob from retries/fragment_retries above, with a default
        # 10ms backoff -- too short for a real scan; raised with real backoff.
        "file_access_retries": 10,
        "retry_sleep_functions": {"file_access": lambda n: min(0.3 * (n + 1), 2.0)},
        # yt-dlp's default silently drops a permanently-failed fragment and
        # still reports success (produced a video minutes shorter than
        # expected with no error). Force an abort instead, so the retry loop
        # below can actually recover rather than accept a silent partial file.
        "skip_unavailable_fragments": False,
    }
    if show_progress:
        options["progress_hooks"] = [_progress]

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rich_info = _extract_rich(job.url, options)
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.process_ie_result(rich_info, download=True)
                path = Path(ydl.prepare_filename(info))
                title = info.get("title") or path.stem

            index = _read_index(dest_dir)
            index[_cache_key(job.url)] = {"filename": path.name, "title": title, "quality": quality}
            _write_index(dest_dir, index)
            return Download(path=path, title=title)

        except yt_dlp.utils.DownloadError as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed, retrying...", file=sys.stderr)
                time.sleep(BACKOFF_SECONDS * attempt)

    raise IngestError(f"Could not download {job.url} after {MAX_ATTEMPTS} attempts: {last_error}")


def probe(path: Path, title: str = "") -> Media:
    """Read fps, duration, dimensions, and frame count from a local media
    file. `frame_count` is a bound for display only -- a specific frame's
    index always comes from that frame's own decoded PTS, not this value."""
    if not path.exists():
        raise IngestError(f"No such media file: {path}")

    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise IngestError(f"No video stream in {path}")

            fps = stream.average_rate or stream.guessed_rate
            if not fps:
                raise IngestError(f"Could not determine frame rate for {path}")

            duration = _duration(container, stream)
            frame_count = stream.frames or int(duration * float(fps))
            return Media(
                path=path,
                title=title or path.stem,
                fps=Fraction(fps),
                frame_count=frame_count,
                duration=duration,
                width=stream.codec_context.width,
                height=stream.codec_context.height,
            )
    except av.FFmpegError as exc:
        raise IngestError(f"Could not read {path}: {exc}") from exc


def _duration(container, stream) -> float:
    """Stream duration, falling back to the container's if the stream lacks one."""
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    raise IngestError("Could not determine duration")
