"""Speech-to-text with word-level timestamps.

`faster-whisper` is a CTranslate2 inference wrapper -- it ships no model
weights. The first `WhisperModel(...)` call for a given size downloads the
converted weights from Hugging Face Hub and caches them under `download_root`;
every call after that loads from the local cache with no network access.
Weights are cached in `data/models/`, not the global HF cache, so they are
visible and inspectable alongside the rest of what this pipeline downloads.

Segment-level timestamps (Whisper's usual unit) are not enough -- the matcher
needs to know exactly which word the target dialogue starts on, so
`word_timestamps=True` is mandatory here, not an optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_MODEL_DIR = Path("data/models")


class TranscribeError(RuntimeError):
    """Raised when transcription cannot be produced."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass(frozen=True)
class Transcript:
    words: list[Word]
    language: str
    language_prob: float

    def joined_text(self) -> str:
        """Words separated by single spaces, in order -- what the matcher
        fuzzy-searches against. Kept separate from `words` so the mapping from
        a character offset back to a specific `Word` stays trivial (each word
        occupies `len(word.text)` characters plus one separating space)."""
        return " ".join(w.text for w in self.words)


def load_model(
    size: str = DEFAULT_MODEL_SIZE,
    device: str = "auto",
    compute_type: str = "auto",
    download_root: Path = DEFAULT_MODEL_DIR,
) -> WhisperModel:
    """Load (downloading and caching on first use) a Whisper model.

    `device="auto"` picks CUDA when available and falls back to CPU, so the
    same call works in this environment (RTX 4060, float16) and elsewhere.
    """
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        return WhisperModel(
            size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
        )
    except Exception as exc:  # model download/load failures are varied; surface plainly
        raise TranscribeError(f"Could not load Whisper model {size!r}: {exc}") from exc


def transcribe(audio_path: Path, model: WhisperModel) -> Transcript:
    """Transcribe `audio_path`, returning a flat, word-level `Transcript`.

    Words without a usable timestamp (rare, but faster-whisper can emit them
    for e.g. non-speech noise) are dropped rather than kept with a fabricated
    time -- a wrong timestamp is worse than a missing word here, since the
    whole pipeline downstream trusts word.start as ground truth.
    """
    if not audio_path.exists():
        raise TranscribeError(f"No such audio file: {audio_path}")

    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            if w.start is None or w.end is None:
                continue
            words.append(Word(text=w.word.strip(), start=w.start, end=w.end, prob=w.probability))

    return Transcript(words=words, language=info.language, language_prob=info.language_probability)
