"""FastAPI backend: submit a (url, dialogue) job, poll its status, fetch results.

Run with:  uv run uvicorn quest1.web.app:app --reload
Then open: http://localhost:8000
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..audio.transcribe import LANGUAGES
from ..ingest.downloader import DEFAULT_QUALITY, QUALITY_CHOICES
from .jobs import Job, manager

app = FastAPI(title="Quest1 - Dialogue Frame Finder")

STATIC_DIR = Path(__file__).parent / "static"

#: Rejected by extension, not by sniffing file contents -- a browser's own
#: file picker/drag-drop already filters to video/* via the frontend's
#: `accept` attribute, so this is a second, server-side check against a
#: request that bypassed that (e.g. a raw API call), not a security boundary.
ALLOWED_UPLOAD_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


class SubmitRequest(BaseModel):
    """Request body for POST /api/jobs."""

    url: str
    dialogue: str
    quality: str = DEFAULT_QUALITY
    language: str | None = None


def _job_view(job: Job) -> dict:
    """The subset of a Job's state that's safe to send to the browser."""
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "result": job.result,
    }


def _clean_language(language: str | None) -> str | None:
    """Blank input means auto-detect; anything else must be one of the
    codes large-v3 actually supports (the web UI offers these as a
    dropdown, so anything else here means a raw API call bypassed it)."""
    if language is None:
        return None
    language = language.strip()
    if not language:
        return None
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unknown language code {language!r}.")
    return language


@app.get("/api/languages")
def list_languages() -> list[dict]:
    """Every language large-v3 supports, for the web UI's dropdown."""
    return [{"code": code, "name": name} for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])]


@app.post("/api/jobs")
def submit_job(req: SubmitRequest) -> dict:
    """Queue a new job for a video URL and return its initial state."""
    if not req.url.strip() or not req.dialogue.strip():
        raise HTTPException(400, "Both a URL and a dialogue are required.")
    if req.quality not in QUALITY_CHOICES:
        raise HTTPException(400, f"quality must be one of {QUALITY_CHOICES}.")
    job = manager.submit(req.url.strip(), req.dialogue.strip(), req.quality, _clean_language(req.language))
    return _job_view(job)


@app.post("/api/jobs/upload")
def submit_upload(
    dialogue: str = Form(...),
    quality: str = Form(DEFAULT_QUALITY),
    language: str = Form(""),
    video: UploadFile = File(...),
) -> dict:
    """Queue a new job for a locally uploaded video file, instead of a URL."""
    if not dialogue.strip():
        raise HTTPException(400, "Dialogue is required.")
    if quality not in QUALITY_CHOICES:
        raise HTTPException(400, f"quality must be one of {QUALITY_CHOICES}.")
    if not video.filename or not video.filename.lower().endswith(ALLOWED_UPLOAD_EXTENSIONS):
        raise HTTPException(400, f"File must be one of: {', '.join(ALLOWED_UPLOAD_EXTENSIONS)}.")

    job = manager.submit_upload(
        video.file, video.filename, dialogue.strip(), quality, _clean_language(language)
    )
    return _job_view(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll a job's current status/stage/result."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return _job_view(job)


@app.get("/api/jobs/{job_id}/image")
def get_job_image(job_id: str) -> FileResponse:
    """Serve the answer frame PNG for a completed job."""
    job = manager.get(job_id)
    if job is None or not job.result or "image_path" not in job.result:
        raise HTTPException(404, "No frame image available for this job.")
    return FileResponse(job.result["image_path"], media_type="image/png")


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    """Serve the downloaded source video for a job, as a download."""
    job = manager.get(job_id)
    if job is None or job.video_path is None or not job.video_path.exists():
        raise HTTPException(404, "No video available for this job.")
    media_type = mimetypes.guess_type(job.video_path.name)[0] or "video/mp4"
    return FileResponse(
        job.video_path,
        media_type=media_type,
        filename=job.video_path.name,
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    """Entry point for `uv run quest1-web`."""
    import uvicorn

    uvicorn.run("quest1.web.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
