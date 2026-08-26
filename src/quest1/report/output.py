"""Render the final result: timestamp, frame number, text, and the frame image.

Matches the output format given in the problem statement:

    Timestamp : HH:MM:SS.sss
    Frame     : <Frame number>
    Text      : "My mind rebels at stagnation"

plus the corresponding image, written to disk since a CLI can't display one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..pipeline import Result
from ..search.matcher import Candidate

DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class Report:
    """What `render()` produces: the printable text plus where the image and
    JSON record were written."""

    text: str
    image_path: Path
    json_path: Path


def render(result: Result, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Report:
    """Write the answer frame and a machine-readable record; return the
    human-readable report text matching the problem statement's format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "answer_frame.png"
    Image.fromarray(result.hit.image).save(image_path)

    record = {
        "timestamp": result.timestamp,
        "frame": result.frame,
        "text": result.match.text,
        "match_score": result.match.score,
        "onset_seconds": result.onset,
        "decoded_pts_seconds": result.hit.pts_time,
        "video_path": str(result.media.path),
        "video_title": result.media.title,
        "fps": float(result.media.fps),
        "image_path": str(image_path),
    }
    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    text = (
        f"Timestamp : {result.timestamp}\n"
        f"Frame     : {result.frame}\n"
        f'Text      : "{result.match.text}"\n'
        f"Image     : {image_path}"
    )
    return Report(text=text, image_path=image_path, json_path=json_path)


def render_not_found(
    dialogue: str,
    threshold: float,
    near_miss: Candidate | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> str:
    """Report for when nothing clears `threshold`: state failure plainly and
    show the best rejected candidate for diagnosis, never a guess presented
    as a confident answer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f'No confident match for "{dialogue}" (threshold={threshold:.0f}).',
    ]
    record: dict = {"dialogue": dialogue, "threshold": threshold, "found": False}

    if near_miss is None:
        lines.append("No candidate of any score was found in the transcript.")
        record["best_candidate"] = None
    else:
        mins, secs = divmod(near_miss.start, 60)
        lines.append(
            f'Best candidate (below threshold, NOT returned as the answer): '
            f'"{near_miss.text}" at {int(mins):02d}:{secs:06.3f}, score={near_miss.score:.1f}'
        )
        record["best_candidate"] = {
            "text": near_miss.text,
            "start": near_miss.start,
            "score": near_miss.score,
        }

    (output_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return "\n".join(lines)
