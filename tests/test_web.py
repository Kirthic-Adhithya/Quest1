from fastapi.testclient import TestClient

from quest1.web.app import app

client = TestClient(app)


def test_submit_rejects_empty_url():
    res = client.post("/api/jobs", json={"url": "  ", "dialogue": "hello"})
    assert res.status_code == 400


def test_submit_rejects_empty_dialogue():
    res = client.post("/api/jobs", json={"url": "https://example.com/v", "dialogue": " "})
    assert res.status_code == 400


def test_get_unknown_job_is_404():
    res = client.get("/api/jobs/does-not-exist")
    assert res.status_code == 404


def test_image_for_unknown_job_is_404():
    res = client.get("/api/jobs/does-not-exist/image")
    assert res.status_code == 404


def test_video_for_unknown_job_is_404():
    res = client.get("/api/jobs/does-not-exist/video")
    assert res.status_code == 404


def test_upload_rejects_empty_dialogue():
    res = client.post(
        "/api/jobs/upload",
        data={"dialogue": " "},
        files={"video": ("clip.mp4", b"not a real video", "video/mp4")},
    )
    assert res.status_code == 400


def test_upload_rejects_invalid_quality():
    res = client.post(
        "/api/jobs/upload",
        data={"dialogue": "hello", "quality": "ultra-hd"},
        files={"video": ("clip.mp4", b"not a real video", "video/mp4")},
    )
    assert res.status_code == 400


def test_upload_rejects_non_video_extension():
    res = client.post(
        "/api/jobs/upload",
        data={"dialogue": "hello"},
        files={"video": ("notes.txt", b"just text", "text/plain")},
    )
    assert res.status_code == 400


def test_upload_rejects_missing_file():
    res = client.post("/api/jobs/upload", data={"dialogue": "hello"})
    assert res.status_code == 422
