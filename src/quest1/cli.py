"""Command-line entry point.

Stage 1 is wired up: the video behind the URL is downloaded (or reused from the
cache) and its properties are displayed. Later stages are added on top.

    uv run quest1 --url https://ok.ru/video/248244667877
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .ingest.downloader import (
    DEFAULT_QUALITY,
    QUALITY_CHOICES,
    IngestError,
    Media,
    fetch,
    probe,
)
from .inputs import InvalidInputError, Job, build_job

DEFAULT_URL = "https://ok.ru/video/248244667877"
DEFAULT_DIALOGUE = "My mind rebels at stagnation"
DEFAULT_MEDIA_DIR = Path("data/media")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quest1",
        description="Find the exact frame in which a dialogue first appears in a video.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Media URL to analyse (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--dialogue",
        default=DEFAULT_DIALOGUE,
        help=f"Dialogue text to look for (default: {DEFAULT_DIALOGUE!r})",
    )
    parser.add_argument(
        "--quality",
        default=DEFAULT_QUALITY,
        choices=QUALITY_CHOICES,
        help=f"Format to download (default: {DEFAULT_QUALITY})",
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=DEFAULT_MEDIA_DIR,
        help=f"Where downloads are cached (default: {DEFAULT_MEDIA_DIR})",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the downloaded video in the system default player",
    )
    parser.add_argument("--version", action="version", version=f"quest1 {__version__}")
    return parser


def show_job(job: Job) -> None:
    print("Quest1 - dialogue frame finder")
    print("=" * 60)
    print(f"URL        : {job.url}")
    print(f"Dialogue   : \"{job.dialogue}\"")
    print("=" * 60)


def show_media(media: Media) -> None:
    print(media.describe())
    print("=" * 60)


def open_in_player(path: Path) -> None:
    """Hand the file to the OS default player. Best effort - never fatal."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - intentional, user asked with --open
        else:
            os.execvp("xdg-open", ["xdg-open", str(path)])
    except OSError as exc:
        print(f"warning: could not open player: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        job = build_job(args.url, args.dialogue)
    except InvalidInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    show_job(job)

    try:
        print("Resolving media...", flush=True)
        download = fetch(job, args.media_dir, args.quality)
        print("Using cached download." if download.from_cache else "Downloaded.", flush=True)
        media = probe(download.path, download.title)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    show_media(media)

    if args.open:
        open_in_player(media.path)

    print("Stage 1 complete. Transcription not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


