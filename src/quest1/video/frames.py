"""Decode the exact video frame on screen at a given onset time.

PyAV, not OpenCV: `cv2.CAP_PROP_POS_FRAMES` seeking is approximate on several
codecs and can silently land on the nearest keyframe. PyAV seeks by
presentation timestamp and decodes forward, so the frame returned is the one
actually asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

#: How far before the target onset to seek. Must comfortably exceed the
#: source's keyframe interval, or the backward seek could land on a keyframe
#: that is itself after the target, forcing an (incorrect) forward search.
SEEK_MARGIN_SECONDS = 2.0
#: Safety bound on how many frames to decode after seeking, in case a seek
#: ever lands somewhere unexpected -- bounds worst-case decode time rather
#: than looping indefinitely.
MAX_FRAMES_AFTER_SEEK = 600


class FrameExtractError(RuntimeError):
    """Raised when the requested frame cannot be located or decoded."""


@dataclass(frozen=True)
class FrameHit:
    """The decoded frame at (or just before) a target onset. `index` and
    `pts_time` come from the decoded frame's own timestamp, not the requested
    onset, so the reported frame and the saved image can never disagree."""

    index: int
    pts_time: float
    image: np.ndarray  # RGB, shape (height, width, 3), uint8


def extract_frame(video_path: Path, onset: float, fps: Fraction) -> FrameHit:
    """The frame on screen at `onset`: the last decoded frame whose own
    timestamp is <= onset (a frame stays on screen until the next one starts).
    """
    if onset < 0:
        raise FrameExtractError(f"onset must be >= 0, got {onset}")

    seek_target = max(0.0, onset - SEEK_MARGIN_SECONDS)

    try:
        with av.open(str(video_path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise FrameExtractError(f"No video stream in {video_path}")

            container.seek(int(seek_target * av.time_base), backward=True, any_frame=False)

            best = None
            for i, frame in enumerate(container.decode(stream)):
                if frame.time is None:
                    continue
                if frame.time <= onset:
                    best = frame
                else:
                    break
                if i >= MAX_FRAMES_AFTER_SEEK:
                    break

            if best is None:
                raise FrameExtractError(
                    f"No frame found at or before {onset:.3f}s in {video_path} "
                    f"(seeked to {seek_target:.3f}s)"
                )

            image = best.to_ndarray(format="rgb24")
            # round() is safe here specifically because best.time is the frame's
            # own timestamp (index/fps exactly, modulo float error) -- rounding
            # an arbitrary onset instead would be wrong (a real bug once: 7799
            # vs the correct 7798), since a frame's window isn't a nearest-point.
            index = round(best.time * float(fps))
            return FrameHit(index=index, pts_time=best.time, image=image)

    except av.FFmpegError as exc:
        raise FrameExtractError(f"Could not decode {video_path}: {exc}") from exc
