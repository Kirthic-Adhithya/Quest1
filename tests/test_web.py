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
