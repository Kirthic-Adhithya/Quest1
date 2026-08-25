"""Stage 4 - forced alignment: pin the exact onset of the first word.

Whisper's word timestamps (stage 2) drift by roughly +/-200ms, because it has
to guess what was said. Forced alignment doesn't guess -- it fits the KNOWN
target text onto the audio using a CTC acoustic model (MMS_FA / Wav2Vec2),
which only has to answer "where does this exact text fall," a far easier
problem. That model outputs one prediction roughly every 20ms, so alignment
is precision to that scale rather than Whisper's ~200ms word boundaries.

We align our target dialogue, not whatever Whisper transcribed -- stage 2's
own wording can drift (observed in practice: "at" heard as "its"), but the
first word, which is what we need the onset of, transcribed correctly in
every run, so aligning the known-correct target text is strictly better than
aligning Whisper's guess.

MMS_FA's vocabulary is 27 unaccented Latin letters -- it has no entries for
Cyrillic, Devanagari, Arabic, CJK, etc., and tokenizing such text raises a
plain KeyError (confirmed directly: Russian input crashes; French, already
Latin-script, does not). The checkpoint is named
`ctc_alignment_mling_uroman` precisely because it expects universal
romanisation as a preprocessing step for non-Latin scripts, so target text is
romanised via the `uroman` package before tokenising here. Latin-script input
romanises to itself, so this adds no behaviour change for English.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
import uroman
from torchaudio.pipelines import MMS_FA as _BUNDLE

from ..search.matcher import normalize

DEFAULT_MODEL_DIR = Path("data/models")
#: Slack added around the coarse (stage-3) word window before alignment, so a
#: Whisper boundary that undershoots the true onset doesn't clip it out of the
#: audio the aligner ever sees.
PADDING_SECONDS = 1.0


class AlignError(RuntimeError):
    """Raised when forced alignment cannot be produced."""


@dataclass(frozen=True)
class Aligner:
    model: torch.nn.Module
    tokenizer: object
    align_fn: object
    device: str
    romanizer: uroman.Uroman


@dataclass(frozen=True)
class WordAlignment:
    text: str
    start: float  # absolute seconds in the source video
    end: float


def load_aligner(model_dir: Path = DEFAULT_MODEL_DIR, device: str | None = None) -> Aligner:
    """Load (downloading and caching on first use) the MMS forced-alignment model.

    Cached under `data/models/torch_hub`, not torch's default global cache, for
    the same reason the Whisper weights are redirected: visible and gitignored
    alongside everything else this pipeline downloads.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_dir / "torch_hub"))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = _BUNDLE.get_model().to(device)
        model.eval()
        tokenizer = _BUNDLE.get_tokenizer()
        align_fn = _BUNDLE.get_aligner()
        romanizer = uroman.Uroman()
    except Exception as exc:  # download/load failures are varied; surface plainly
        raise AlignError(f"Could not load forced-alignment model: {exc}") from exc

    return Aligner(model=model, tokenizer=tokenizer, align_fn=align_fn, device=device, romanizer=romanizer)


def _decode_clip(video_path: Path, start: float, end: float) -> torch.Tensor:
    """Decode [start, end) of `video_path`'s audio to a 16 kHz mono float32 tensor.

    Kept in-memory rather than writing a temp WAV, since alignment clips are a
    few seconds -- decoding straight to a tensor avoids a disk round-trip for
    something this small.
    """
    try:
        with av.open(str(video_path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AlignError(f"No audio stream in {video_path}")

            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=_BUNDLE.sample_rate
            )
            container.seek(int(max(0.0, start) * av.time_base))

            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                if frame.time is None or frame.time < start:
                    continue
                if frame.time > end:
                    break
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())
    except av.FFmpegError as exc:
        raise AlignError(f"Could not decode audio from {video_path}: {exc}") from exc

    if not chunks:
        raise AlignError(f"No audio decoded in [{start}, {end}) from {video_path}")

    data = np.concatenate(chunks, axis=1)
    return torch.from_numpy(data.astype(np.float32) / 32768.0)


def align_words(
    video_path: Path,
    target_text: str,
    window_start: float,
    window_end: float,
    aligner: Aligner,
    padding: float = PADDING_SECONDS,
) -> list[WordAlignment]:
    """Fit `target_text` onto the audio in [window_start, window_end] (+padding),
    returning one absolute-time span per word, in order.
    """
    words = normalize(target_text).split()
    if not words:
        raise AlignError("Target text has no words to align.")

    clip_start = max(0.0, window_start - padding)
    clip_end = window_end + padding
    waveform = _decode_clip(video_path, clip_start, clip_end)

    # Romanised per word, not as one joined string, so the token count stays
    # 1:1 with `words` -- romanising a whole sentence at once can re-tokenise
    # scripts into a different number of space-separated pieces than the
    # original word boundaries.
    romanized_words = [
        aligner.romanizer.romanize_string(w).replace(" ", "").lower() or w
        for w in words
    ]

    with torch.inference_mode():
        emission, _ = aligner.model(waveform.to(aligner.device))

    tokens = aligner.tokenizer(romanized_words)
    token_spans = aligner.align_fn(emission[0], tokens)

    num_frames = emission.shape[1]
    seconds_per_frame = waveform.shape[1] / num_frames / _BUNDLE.sample_rate

    return [
        WordAlignment(
            text=word,
            start=clip_start + spans[0].start * seconds_per_frame,
            end=clip_start + spans[-1].end * seconds_per_frame,
        )
        for word, spans in zip(words, token_spans)
    ]


def refine_onset(
    video_path: Path,
    target_text: str,
    window_start: float,
    window_end: float,
    aligner: Aligner,
    padding: float = PADDING_SECONDS,
) -> float:
    """The onset (absolute seconds) of the first word of `target_text` --
    the answer frame's defining instant, per DESIGN.md's convention."""
    words = align_words(video_path, target_text, window_start, window_end, aligner, padding)
    return words[0].start
