"""Orchestrator: wires ingest -> transcribe -> match -> ocr -> report.

Only ingest -> transcribe exists so far; match/align/locate/report are added as
those stages are built.
"""

from __future__ import annotations

from pathlib import Path

from .audio.extract import extract_audio
from .audio.transcribe import DEFAULT_MODEL_DIR, DEFAULT_MODEL_SIZE, Transcript, load_model, transcribe
from .ingest.downloader import DEFAULT_QUALITY, Media, fetch, probe
from .inputs import Job


def run_transcription(
    job: Job,
    media_dir: Path = Path("data/media"),
    quality: str = DEFAULT_QUALITY,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[Media, Transcript]:
    """Stages 1-2: get the media locally, then transcribe its audio.

    Each stage's own cache (download cache, decoded-audio cache) means a
    re-run of this function during development only re-does whatever changed.
    """
    download = fetch(job, media_dir, quality)
    media = probe(download.path, download.title)

    audio_path = media_dir / f"{media.path.stem}.wav"
    audio = extract_audio(media.path, audio_path)

    model = load_model(model_size, download_root=model_dir)
    transcript = transcribe(audio, model)

    return media, transcript
