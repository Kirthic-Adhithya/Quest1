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

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def _register_cuda_dll_dirs() -> None:
    """Point CTranslate2 (faster-whisper's backend) at torch's bundled
    cuBLAS/cuDNN on Windows, instead of installing a separate ~2GB copy of
    the same libraries. Must run before `faster_whisper` is imported, since
    CTranslate2's loader only honours PATH, not `os.add_dll_directory`."""
    if sys.platform != "win32":
        return
    torch_lib = str(Path(torch.__file__).parent / "lib")
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 -- must follow the PATH fix above

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_MODEL_DIR = Path("data/models")

_model_cache: dict[tuple, WhisperModel] = {}


class TranscribeError(RuntimeError):
    """Raised when transcription cannot be produced."""


@dataclass(frozen=True)
class Word:
    """One transcribed word with its timing and confidence."""

    text: str
    start: float
    end: float
    prob: float


@dataclass(frozen=True)
class Transcript:
    """A full transcription: every word, in order, plus detected language."""

    words: list[Word]
    language: str
    language_prob: float

    def to_json(self) -> str:
        """Serialize for the on-disk transcript cache."""
        return json.dumps(
            {
                "language": self.language,
                "language_prob": self.language_prob,
                "words": [
                    {"text": w.text, "start": w.start, "end": w.end, "prob": w.prob}
                    for w in self.words
                ],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "Transcript":
        """Deserialize a cached transcript."""
        data = json.loads(raw)
        return cls(
            words=[Word(**w) for w in data["words"]],
            language=data["language"],
            language_prob=data["language_prob"],
        )


def load_model(
    size: str = DEFAULT_MODEL_SIZE,
    device: str = "auto",
    compute_type: str = "auto",
    download_root: Path = DEFAULT_MODEL_DIR,
) -> WhisperModel:
    """Load (downloading and caching to disk on first use) a Whisper model.

    `device="auto"` picks CUDA when available and falls back to CPU, so the
    same call works in this environment (RTX 4060, float16) and elsewhere.

    Also cached in-process (measured: ~10s to load `large-v3` onto the GPU),
    keyed by every argument here -- a caller handling several jobs in one run
    (the web app) only pays that cost once instead of once per job.
    """
    key = (size, device, compute_type, str(download_root))
    if key not in _model_cache:
        download_root.mkdir(parents=True, exist_ok=True)
        try:
            _model_cache[key] = WhisperModel(
                size,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root),
            )
        except Exception as exc:  # model download/load failures are varied; surface plainly
            raise TranscribeError(f"Could not load Whisper model {size!r}: {exc}") from exc
    return _model_cache[key]


def transcribe(audio_path: Path, model: WhisperModel, language: str | None = None) -> Transcript:
    """Transcribe `audio_path` into a flat, word-level `Transcript`.

    `language=None` auto-detects from the first ~30s of audio, which can
    mis-detect on a non-speech opening (title music); pass the language
    explicitly when it's known to avoid that. Words with no usable timestamp
    are dropped rather than kept with a fabricated one, since downstream
    code trusts word.start as ground truth.
    """
    if not audio_path.exists():
        raise TranscribeError(f"No such audio file: {audio_path}")

    segments, info = model.transcribe(str(audio_path), word_timestamps=True, language=language)

    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            if w.start is None or w.end is None:
                continue
            words.append(Word(text=w.word.strip(), start=w.start, end=w.end, prob=w.probability))

    return Transcript(words=words, language=info.language, language_prob=info.language_probability)
