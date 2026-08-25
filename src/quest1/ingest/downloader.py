"""Stage 1 - fetch the media behind the URL and probe its properties.

Two responsibilities, kept separate so either can be used alone:

* `fetch`  - resolve a URL to a local file via yt-dlp, caching by video id.
* `probe`  - read fps / frame count / duration / size out of the local file.

Everything downstream converts between time and frame numbers using the `fps`
reported here, so it is a `Fraction` (e.g. 25/1, 30000/1001) rather than a float.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
import yt_dlp

from ..inputs import Job

#: ok.ru exposes named rungs rather than numeric heights; yt-dlp maps these for
#: other sites too, falling back to "best available at or below this quality".
QUALITY_CHOICES = ("mobile", "lowest", "low", "sd", "hd", "best")
DEFAULT_QUALITY = "sd"

#: Height-bounded selectors, ranked purely by real per-format resolution
#: metadata -- never by a site's own literal format-id names.
#:
#: ok.ru's named tiers ("sd", "hd", ...) carry no resolution or bitrate at all
#: (see DESIGN.md), so preferring them by name ahead of ranked formats is
#: actively wrong: it is unranked data racing against ranked data, and which
#: one "wins" depends on which alternative in the selector string happens to
#: match first. Height-bounded ranking with no literal-name special case is
#: deterministic on every site.
#:
#: Modern YouTube (and many other sites) serve DASH: video-only and audio-only
#: streams with no single muxed format at all. A selector like "best[height<=480]"
#: then matches nothing and download fails outright. Each tier therefore tries a
#: separate video+audio pair first (merged by ffmpeg) and only falls back to a
#: single muxed format for sites -- ok.ru included -- that offer one instead.
_GENERIC_FALLBACK = {
    "mobile": "worstvideo[height<=240]+worstaudio/worst[height<=240]/worst",
    "lowest": "worstvideo+worstaudio/worst",
    "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "sd": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "hd": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    "best": "bestvideo+bestaudio/best",
}


def _format_selector(quality: str) -> str:
    return _GENERIC_FALLBACK[quality]

#: ok.ru intermittently resets connections mid-handshake, so extraction is
#: retried rather than assumed to succeed on the first attempt.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 2.0

#: The Odnoklassniki extractor tries a rich "desktop" page first (full HLS
#: ladder, up to 2160p, with real resolution per variant) and silently falls
#: back to a crippled "mobile" page (5 named tiers, no resolution/bitrate at
#: all) on *any* error from the desktop attempt -- which our flaky host
#: triggers often. The fallback succeeds without raising, so the ordinary
#: retry loop never sees it as a failure. Metadata extraction is therefore
#: retried on its own until the rich ladder is observed.
RICH_PROBE_ATTEMPTS = 6
RICH_PROBE_BACKOFF = 1.5

#: Maps URL -> downloaded filename, so a repeat run never touches the network.
#: yt-dlp alone would still re-extract the page just to learn the output filename.
CACHE_INDEX = ".cache.json"

CR = "\r"


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
        return self.path.stat().st_size

    def frame_at(self, seconds: float) -> int:
        """Nominal frame index containing `seconds` -- a seek hint, not the
        final answer (that comes from the decoded frame's own PTS; see
        `video/frames.py`).

        Deliberately `floor`, not `round`: frame N is on screen for
        `[N/fps, (N+1)/fps)`, a containment question, not a nearest-neighbour
        one. `round` was tried first and produced a real off-by-one -- for a
        real timestamp on the reference video, `round` gave frame 7799 while
        the actually-decoded frame at that instant was 7798, because the
        timestamp's fractional frame position (.535) was past round()'s .5
        threshold despite still falling inside frame 7798's window.
        """
        return int(seconds * float(self.fps))

    def describe(self) -> str:
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
    try:
        return json.loads((dest_dir / CACHE_INDEX).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_index(dest_dir: Path, index: dict) -> None:
    try:
        (dest_dir / CACHE_INDEX).write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # a broken cache must never break the pipeline
        print(f"warning: could not write cache index: {exc}", file=sys.stderr)


def lookup_cached(dest_dir: Path, url: str, quality: str) -> Download | None:
    """Return a previously downloaded file for `url` at `quality`, or None.

    Quality is part of the cache key deliberately: a file fetched at "sd" must
    never be handed back for a "hd" request. An old entry recorded before this
    field existed (no "quality" key) is treated as a miss so it gets replaced
    rather than silently trusted.
    """
    entry = _read_index(dest_dir).get(url)
    if not entry or entry.get("quality") != quality:
        return None
    path = dest_dir / entry["filename"]
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return Download(path=path, title=entry.get("title", path.stem), from_cache=True)


def _evict_stale(dest_dir: Path, url: str) -> None:
    """Delete a previously downloaded file for `url`, plus any leftover partial
    download state (`.part`, `.part-FragN.part`, `.ytdl`) sharing its stem.

    yt-dlp's own outtmpl is quality-agnostic (`%(id)s.%(ext)s`), so without this
    a re-download would either be skipped (yt-dlp defaults to not overwriting an
    existing file) or resume from partial-download state left by a *different*
    selection than the one about to be attempted -- exactly what produced an
    unfixed raw MPEG-TS file wearing an `.mp4` extension in practice. A fresh
    attempt always starts from nothing.
    """
    entry = _read_index(dest_dir).get(url)
    if not entry:
        return
    stale = dest_dir / entry["filename"]
    for leftover in dest_dir.glob(f"{stale.stem}.*"):
        if leftover != dest_dir / CACHE_INDEX:
            leftover.unlink()


def _evict_other_videos(dest_dir: Path, keep_url: str) -> None:
    """Delete every cached video except `keep_url`, plus each one's audio,
    transcripts, and any partial-download leftovers sharing its filename stem.

    The cache originally kept every video ever fetched, so switching back and
    forth between videos during development never re-paid a download. That
    grows disk usage without bound, which is the wrong default for a deployed
    instance meant to process one video at a time -- each `fetch()` call now
    prunes the cache down to just the video currently being requested.
    """
    index = _read_index(dest_dir)
    changed = False
    for url, entry in list(index.items()):
        if url == keep_url:
            continue
        stale = dest_dir / entry["filename"]
        for leftover in dest_dir.glob(f"{stale.stem}.*"):
            if leftover != dest_dir / CACHE_INDEX:
                leftover.unlink(missing_ok=True)
        del index[url]
        changed = True
    if changed:
        _write_index(dest_dir, index)


def _has_resolution_ladder(info: dict) -> bool:
    """True once the extracted metadata carries per-format resolution.

    ok.ru's fallback "mobile" formats never report height/tbr at all, so a
    format selector like "bestvideo+bestaudio" has nothing to rank by and picks
    close to arbitrarily. The richer "desktop" path resolves the HLS manifest
    into per-variant formats with a real width/height, which is what makes a
    quality selector meaningful in the first place.
    """
    return any(f.get("height") for f in info.get("formats", []))


def _extract_rich(url: str, options: dict) -> dict:
    """Extract metadata, retrying until a real resolution ladder is present.

    A retry here is deliberately separate from `fetch`'s own retry loop: the
    mobile fallback is not an exception, it is a successful-looking result with
    worse data, so it would never trigger an ordinary error-triggered retry.
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
        if best_seen is None or len(info.get("formats", [])) > len(
            best_seen.get("formats", [])
        ):
            best_seen = info
        if attempt < RICH_PROBE_ATTEMPTS:
            time.sleep(RICH_PROBE_BACKOFF * attempt)

    if best_seen is None:
        raise IngestError(f"Could not read any format metadata for {url}")

    print(
        "warning: ok.ru only offered its limited mobile format list after "
        f"{RICH_PROBE_ATTEMPTS} attempts (no per-format resolution data); "
        "quality selection may not reflect the true best available.",
        file=sys.stderr,
    )
    return best_seen


def _progress(status: dict) -> None:
    """Single-line download progress, so a 50-minute episode is not a silent wait."""
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
    """Download the media for `job` into `dest_dir`, reusing an existing copy.

    Only one video is ever kept in `dest_dir` at a time: every call evicts any
    *other* cached video first (`_evict_other_videos`), so switching to a new
    URL frees the previous video's download, audio, and transcripts rather
    than accumulating them -- the right default for a deployed instance that
    processes one video at a time, at the cost of needing to re-download if
    you switch back and forth between videos during development.

    A cached (url, quality) pair for the *current* URL still short-circuits
    before any network call, which matters because the host is flaky:
    re-running the pipeline during development should not depend on ok.ru
    being reachable at that moment. Requesting a different quality for the
    same URL is *not* a cache hit -- the stale file is deleted and a fresh
    download replaces it.

    Connection resets from the host are retried with a linear backoff; only the
    final failure is surfaced.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    _evict_other_videos(dest_dir, job.url)

    cached = lookup_cached(dest_dir, job.url, quality)
    if cached is not None:
        return cached

    _evict_stale(dest_dir, job.url)

    options = {
        "format": _format_selector(quality),
        # Video+audio pairs need muxing; yt-dlp does not discover the ffmpeg
        # bundled by imageio-ffmpeg on its own.
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "merge_output_format": "mp4",
        # A single HLS format downloads as raw, unfixed MPEG-TS wearing an
        # ".mp4" extension unless explicitly remuxed -- PyAV opens it (h264/aac
        # decode fine) but the container carries no duration, which is exactly
        # what broke `probe()` in practice. Force a real container regardless
        # of how yt-dlp fetched the stream.
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        # The host drops connections mid-transfer; chunking plus resume means a
        # drop costs one chunk rather than the whole 50-minute file.
        "continuedl": True,
        "http_chunk_size": 10 * 1024 * 1024,
        # On Windows, an HLS download writes and renames one small file per
        # fragment. A real-time antivirus scanner (confirmed present: Windows
        # Defender) can hold a brief lock on a just-written fragment file,
        # racing yt-dlp's rename -- yt-dlp's *own* retry knob for this class
        # of error is `file_access_retries` (default 3), completely separate
        # from `retries`/`fragment_retries` above, and its default backoff
        # between attempts is 10ms -- far shorter than a typical AV scan.
        # Confirmed via yt-dlp's own source, not guessed. More attempts with a
        # real backoff gives the lock time to clear instead of giving up fast.
        "file_access_retries": 10,
        "retry_sleep_functions": {"file_access": lambda n: min(0.3 * (n + 1), 2.0)},
        # yt-dlp's default (skip_unavailable_fragments=True) silently drops any
        # fragment that permanently fails and still reports success -- found in
        # practice: a real run produced a video 2 minutes shorter than every
        # previous verified download, with no error, no warning, exit code 0.
        # For a tool whose entire purpose is exact-frame correctness, a silent
        # partial download is worse than a loud failure: force an abort (which
        # raises DownloadError, already retried by the loop below) instead.
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
            index[job.url] = {"filename": path.name, "title": title, "quality": quality}
            _write_index(dest_dir, index)
            return Download(path=path, title=title)

        except yt_dlp.utils.DownloadError as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(
                    f"  attempt {attempt}/{MAX_ATTEMPTS} failed, retrying...",
                    file=sys.stderr,
                )
                time.sleep(BACKOFF_SECONDS * attempt)

    raise IngestError(
        f"Could not download {job.url} after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def probe(path: Path, title: str = "") -> Media:
    """Read stream properties out of a local media file.

    `frames` is absent from some containers, so fall back to duration * fps.
    Downstream code treats frame_count as a bound, never as ground truth for a
    specific frame -- that always comes from the decoded frame's own PTS.
    """
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
    """Prefer the stream's own duration, falling back to the container's."""
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    raise IngestError("Could not determine duration")
