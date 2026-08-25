import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from quest1.ingest.downloader import Media
from quest1.pipeline import Result
from quest1.report.output import render, render_not_found
from quest1.search.matcher import Candidate
from quest1.video.frames import FrameHit


def _result(tmp_path: Path) -> Result:
    media = Media(
        path=tmp_path / "video.mp4",
        title="Test Video",
        fps=Fraction(24, 1),
        frame_count=1000,
        duration=41.7,
        width=64,
        height=48,
    )
    match = Candidate(
        word_start=10, word_end=15, start=5.0, end=6.0,
        text="my mind rebels at stagnation", score=94.7,
    )
    hit = FrameHit(index=120, pts_time=5.0, image=np.zeros((48, 64, 3), dtype=np.uint8))
    return Result(media=media, match=match, onset=5.01, hit=hit)


def test_render_writes_image_and_json_matching_result(tmp_path):
    result = _result(tmp_path)
    report = render(result, output_dir=tmp_path / "out")

    assert report.image_path.exists()
    assert report.json_path.exists()

    record = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert record["frame"] == 120
    assert record["text"] == "my mind rebels at stagnation"
    assert record["timestamp"] == result.timestamp


def test_render_text_matches_problem_statement_format(tmp_path):
    result = _result(tmp_path)
    report = render(result, output_dir=tmp_path / "out")

    assert "Timestamp :" in report.text
    assert "Frame     :" in report.text
    assert 'Text      : "my mind rebels at stagnation"' in report.text


def test_render_not_found_labels_near_miss_as_rejected(tmp_path):
    near_miss = Candidate(
        word_start=0, word_end=3, start=100.0, end=101.0, text="unrelated words here", score=42.0
    )
    text = render_not_found("My mind rebels at stagnation", 70.0, near_miss, output_dir=tmp_path)
    assert "No confident match" in text
    assert "NOT returned as the answer" in text
    assert "42.0" in text


def test_render_not_found_handles_no_candidates_at_all(tmp_path):
    text = render_not_found("My mind rebels at stagnation", 70.0, None, output_dir=tmp_path)
    assert "No confident match" in text
    assert "No candidate of any score" in text
