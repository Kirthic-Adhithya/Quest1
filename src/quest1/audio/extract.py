"""Extract a video's audio track to a 16 kHz mono WAV file for ASR.

Decoded with PyAV rather than shelling out to ffmpeg: PyAV already links the
ffmpeg libraries (it is a dependency of the ingest stage), so this avoids a
second process and a second place to wire up `ffmpeg_location`.

16 kHz mono is not an arbitrary choice -- it is the sample rate Whisper's
encoder was trained on, so resampling here means transcription never has to
think about the source video's actual audio format.
"""

from __future__ import annotations

from pathlib import Path

import av

WHISPER_SAMPLE_RATE = 16_000


class AudioExtractError(RuntimeError):
    """Raised when a video's audio track cannot be read or decoded."""


def extract_audio(video_path: Path, dest_path: Path) -> Path:
    """Decode the audio stream of `video_path` to 16 kHz mono PCM WAV.

    Cached like the download itself: if `dest_path` already exists and is
    non-empty, it is reused rather than re-decoded, since decoding a
    54-minute episode is themselves a real cost worth avoiding on re-runs.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with av.open(str(video_path)) as in_container:
            in_stream = next(
                (s for s in in_container.streams if s.type == "audio"), None
            )
            if in_stream is None:
                raise AudioExtractError(f"No audio stream in {video_path}")

            with av.open(str(dest_path), mode="w") as out_container:
                out_stream = out_container.add_stream("pcm_s16le", rate=WHISPER_SAMPLE_RATE)
                out_stream.layout = "mono"

                resampler = av.AudioResampler(
                    format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE
                )

                for frame in in_container.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        for packet in out_stream.encode(resampled):
                            out_container.mux(packet)

                for packet in out_stream.encode(None):  # flush
                    out_container.mux(packet)

    except av.FFmpegError as exc:
        dest_path.unlink(missing_ok=True)
        raise AudioExtractError(f"Could not extract audio from {video_path}: {exc}") from exc

    return dest_path
