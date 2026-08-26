"""In-memory job queue running the pipeline one video at a time.

A single background worker thread processes jobs from a queue, never more
than one concurrently. This is a direct consequence of a real incident: this
pipeline was measured to need ~4GB of GPU VRAM per Whisper instance, and
running three at once on an 8GB card produced near-total stalling rather than
any error (see APPROACH.md). A web server makes concurrent submission trivial
to trigger by accident -- two tabs, a refresh mid-run -- so jobs are queued
and run strictly sequentially rather than firing a thread per request.

State is in-memory only (a plain dict), which is deliberate: this is a
localhost, single-user tool, not a multi-instance service, so there is
nothing to gain from a real job store and a real cost (a new dependency, a
new failure mode) in adding one.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from ..audio.align import AlignError, load_aligner, refine_onset
from ..audio.extract import AudioExtractError
from ..audio.transcribe import TranscribeError, pick_model_size
from ..ingest.downloader import DEFAULT_QUALITY, IngestError, probe
from ..inputs import InvalidInputError, build_job, parse_dialogue
from ..pipeline import Result, run_transcription, transcribe_media
from ..report.output import render, render_not_found
from ..search.matcher import DEFAULT_THRESHOLD, best_match, best_near_miss
from ..video.frames import FrameExtractError, extract_frame

JobStatus = Literal["queued", "running", "done", "error", "not_found"]

MEDIA_DIR = Path("data/media")
UPLOAD_DIR = Path("data/media/uploads")
OUTPUT_ROOT = Path("outputs/web")


@dataclass
class Job:
    """One submitted request and its current state, polled by the frontend.

    Either `url` names something to download, or `upload_path` already
    points at a local file -- never both meaningfully at once. Exactly one
    of the two source paths in `_process` applies, chosen by whether
    `upload_path` is set.
    """

    id: str
    url: str
    dialogue: str
    quality: str = DEFAULT_QUALITY
    language: str | None = None
    upload_path: Path | None = None
    status: JobStatus = "queued"
    stage: str = "Queued"
    error: str | None = None
    result: dict | None = None
    video_path: Path | None = None
    created_at: float = field(default_factory=time.time)


class JobManager:
    """Owns the job dict and the single worker thread that processes them."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _enqueue(self, job: Job) -> Job:
        """Register a job and hand it to the worker thread."""
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def submit(
        self, url: str, dialogue: str, quality: str = DEFAULT_QUALITY, language: str | None = None
    ) -> Job:
        """Create a job for a video at `url` and enqueue it."""
        job = Job(id=uuid.uuid4().hex[:12], url=url, dialogue=dialogue, quality=quality, language=language)
        return self._enqueue(job)

    def submit_upload(
        self,
        file_obj: BinaryIO,
        filename: str,
        dialogue: str,
        quality: str = DEFAULT_QUALITY,
        language: str | None = None,
    ) -> Job:
        """Save an uploaded video to disk and enqueue a job that skips the
        download step, probing the local file directly. Evicts any
        previous upload first -- one at a time, matching the download
        cache's single-video disk-usage policy."""
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        job_id = uuid.uuid4().hex[:12]
        dest_dir = UPLOAD_DIR / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(file_obj, out)

        job = Job(
            id=job_id, url=f"(uploaded) {filename}", dialogue=dialogue,
            quality=quality, language=language, upload_path=dest_path,
        )
        return self._enqueue(job)

    def get(self, job_id: str) -> Job | None:
        """Look up a job by id, for polling."""
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job: Job, **updates) -> None:
        """Update a job's fields under the lock."""
        with self._lock:
            for key, value in updates.items():
                setattr(job, key, value)

    def _run_worker(self) -> None:
        """Pull one job at a time off the queue and run the pipeline on it,
        recording any failure onto the job instead of raising."""
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except (
                InvalidInputError, IngestError, AudioExtractError,
                TranscribeError, AlignError, FrameExtractError,
            ) as exc:
                self._set(job, status="error", error=str(exc), stage="Failed")
            except Exception as exc:  # a stuck worker thread is worse than a vague error
                self._set(job, status="error", error=f"Unexpected error: {exc}", stage="Failed")

    def _process(self, job: Job) -> None:
        """Run the full pipeline for one job: transcribe, match, align,
        extract the frame, and record the result (or a not-found diagnosis)
        onto the job."""
        self._set(job, status="running", stage="Validating input")
        model_size = pick_model_size(job.language)

        if job.upload_path is not None:
            dialogue = parse_dialogue(job.dialogue)
            self._set(job, stage="Probing the uploaded video")
            media = probe(job.upload_path, title=job.upload_path.stem)
            job.video_path = media.path
            self._set(job, stage="Transcribing audio (first run can take several minutes)")
            transcript = transcribe_media(media, MEDIA_DIR, model_size, language=job.language)
        else:
            parsed = build_job(job.url, job.dialogue)
            dialogue = parsed.dialogue
            self._set(job, stage="Downloading media and transcribing audio (first run can take several minutes)")
            media, transcript = run_transcription(
                parsed, MEDIA_DIR, job.quality, model_size, language=job.language
            )
            job.video_path = media.path

        self._set(job, stage="Matching dialogue against the transcript")
        match = best_match(transcript, dialogue, DEFAULT_THRESHOLD)

        output_dir = OUTPUT_ROOT / job.id
        if match is None:
            near_miss = best_near_miss(transcript, dialogue)
            render_not_found(dialogue, DEFAULT_THRESHOLD, near_miss, output_dir)
            near_miss_dict = None
            if near_miss is not None:
                mins, secs = divmod(near_miss.start, 60)
                near_miss_dict = {
                    "text": near_miss.text,
                    "score": round(near_miss.score, 1),
                    "timestamp": f"{int(mins):02d}:{secs:06.3f}",
                }
            self._set(
                job,
                status="not_found",
                stage="Not found",
                result={"threshold": DEFAULT_THRESHOLD, "near_miss": near_miss_dict},
            )
            return

        self._set(job, stage="Refining onset via forced alignment")
        aligner = load_aligner()
        onset = refine_onset(media.path, dialogue, match.start, match.end, aligner)

        self._set(job, stage="Extracting the answer frame")
        hit = extract_frame(media.path, onset, media.fps)

        result = Result(media=media, match=match, onset=onset, hit=hit)
        report = render(result, output_dir)

        self._set(
            job,
            status="done",
            stage="Done",
            result={
                "timestamp": result.timestamp,
                "seconds": result.hit.pts_time,
                "frame": result.frame,
                "text": result.match.text,
                "score": round(result.match.score, 1),
                "video_title": result.media.title,
                "image_path": str(report.image_path),
            },
        )


manager = JobManager()
