"""Command-line entry point.

Runs the full pipeline: download -> transcribe -> match -> align -> extract
frame -> report. Each stage prints progress, since transcription and alignment
alone can take several minutes on a long video and a silent multi-minute CLI
run reads as broken.

    uv run quest1 --url https://ok.ru/video/248244667877 --dialogue "My mind rebels at stagnation"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .audio.align import DEFAULT_MODEL_DIR as ALIGN_MODEL_DIR
from .audio.align import PADDING_SECONDS, load_aligner, refine_onset
from .audio.transcribe import DEFAULT_MODEL_DIR, DEFAULT_MODEL_SIZE
from .ingest.downloader import DEFAULT_QUALITY, QUALITY_CHOICES, IngestError
from .inputs import InvalidInputError, Job, build_job
from .pipeline import Result, run_transcription
from .report.output import DEFAULT_OUTPUT_DIR, render, render_not_found
from .search.matcher import DEFAULT_THRESHOLD, best_match, find_candidates
from .video.frames import FrameExtractError, extract_frame

DEFAULT_URL = "https://ok.ru/video/248244667877"
DEFAULT_DIALOGUE = "My mind rebels at stagnation"
DEFAULT_MEDIA_DIR = Path("data/media")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quest1",
        description="Find the exact frame in which a dialogue first appears in a video.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Media URL to analyse (default: {DEFAULT_URL})")
    parser.add_argument(
        "--dialogue",
        default=DEFAULT_DIALOGUE,
        help=f"Dialogue text to look for (default: {DEFAULT_DIALOGUE!r})",
    )
    parser.add_argument(
        "--quality", default=DEFAULT_QUALITY, choices=QUALITY_CHOICES,
        help=f"Format to download (default: {DEFAULT_QUALITY})",
    )
    parser.add_argument(
        "--language", default=None,
        help="Force the transcription language (e.g. 'en'). Default: auto-detect. "
             "Auto-detect can mis-fire on a non-speech opening (see DESIGN.md); "
             "pass this explicitly when the language is known.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum fuzzy-match score (0-100) to accept a candidate (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--model-size", default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE}); smaller is faster for dev iteration",
    )
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR, help=f"Download/cache dir (default: {DEFAULT_MEDIA_DIR})")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=f"Whisper model cache dir (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--align-model-dir", type=Path, default=ALIGN_MODEL_DIR, help=f"Alignment model cache dir (default: {ALIGN_MODEL_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Where the report and frame image are written (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--open", action="store_true", help="Open the answer frame image after a successful run")
    parser.add_argument("--version", action="version", version=f"quest1 {__version__}")
    return parser


def show_job(job: Job) -> None:
    print("Quest1 - dialogue frame finder")
    print("=" * 60)
    print(f'URL        : {job.url}')
    print(f'Dialogue   : "{job.dialogue}"')
    print("=" * 60)


def open_file(path: Path) -> None:
    """Hand a file to the OS default viewer. Best effort - never fatal."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - intentional, user asked with --open
        else:
            os.execvp("xdg-open", ["xdg-open", str(path)])
    except OSError as exc:
        print(f"warning: could not open {path}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        job = build_job(args.url, args.dialogue)
    except InvalidInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    show_job(job)

    try:
        print("Stage 1-2: downloading media and transcribing audio "
              "(a long video can take several minutes on first run)...", flush=True)
        media, transcript = run_transcription(
            job, args.media_dir, args.quality, args.model_size, args.model_dir, args.language
        )
        print(media.describe())
        print(f"Transcribed {len(transcript.words)} words "
              f"(language={transcript.language}, confidence={transcript.language_prob:.2f}).")

        print("Stage 3: matching dialogue against the transcript...", flush=True)
        match = best_match(transcript, job.dialogue, args.threshold)

        if match is None:
            near_miss = None
            all_candidates = find_candidates(transcript, job.dialogue, threshold=0.0)
            if all_candidates:
                near_miss = max(all_candidates, key=lambda c: c.score)
            print(render_not_found(job.dialogue, args.threshold, near_miss, args.output_dir))
            return 1

        print(f'Matched "{match.text}" (score={match.score:.1f}) at ~{match.start:.2f}s.')

        print("Stage 4: refining onset via forced alignment...", flush=True)
        aligner = load_aligner(args.align_model_dir)
        onset = refine_onset(media.path, job.dialogue, match.start, match.end, aligner, PADDING_SECONDS)

        print("Stage 5: extracting the answer frame...", flush=True)
        hit = extract_frame(media.path, onset, media.fps)

        result = Result(media=media, match=match, onset=onset, hit=hit)

    except (IngestError, FrameExtractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    report = render(result, args.output_dir)
    print(report.text)
    print("=" * 60)

    if args.open:
        open_file(report.image_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
