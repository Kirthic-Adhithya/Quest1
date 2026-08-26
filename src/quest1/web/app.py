"""FastAPI backend: submit a (url, dialogue) job, poll its status, fetch results.

Run with:  uv run uvicorn quest1.web.app:app --reload
Then open: http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ingest.downloader import DEFAULT_QUALITY, QUALITY_CHOICES
from .jobs import Job, manager

app = FastAPI(title="Quest1 - Dialogue Frame Finder")

STATIC_DIR = Path(__file__).parent / "static"


class SubmitRequest(BaseModel):
    """Request body for POST /api/jobs."""

    url: str
    dialogue: str
    quality: str = DEFAULT_QUALITY


def _job_view(job: Job) -> dict:
    """The subset of a Job's state that's safe to send to the browser."""
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "result": job.result,
    }


@app.post("/api/jobs")
def submit_job(req: SubmitRequest) -> dict:
    """Queue a new job and return its initial state."""
    if not req.url.strip() or not req.dialogue.strip():
        raise HTTPException(400, "Both a URL and a dialogue are required.")
    if req.quality not in QUALITY_CHOICES:
        raise HTTPException(400, f"quality must be one of {QUALITY_CHOICES}.")
    job = manager.submit(req.url.strip(), req.dialogue.strip(), req.quality)
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
    return FileResponse(
        job.video_path,
        media_type="video/mp4",
        filename=job.video_path.name,
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    """Entry point for `uv run quest1-web`."""
    import uvicorn

    uvicorn.run("quest1.web.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
