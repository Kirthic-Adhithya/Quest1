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
    """Make torch's bundled cuBLAS/cuDNN discoverable to CTranslate2 on Windows.

    CTranslate2 (faster-whisper's backend) needs its own cuBLAS 12 / cuDNN 9
    on its DLL search path. It does not need its *own copy* of them, though --
    `torch` (a hard dependency of this project regardless, for stage 4's
    forced alignment) already bundles equivalent DLLs in `torch/lib`. This
    used to install a separate `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip
    package pair for the same DLLs -- ~2 GB of exact duplication with what
    `torch` already ships, confirmed by inspecting `torch/lib` directly.
    Pointing at torch's copies instead (verified: a real CUDA transcription
    run succeeds using only these) removes that duplication entirely.

    `os.add_dll_directory` alone is not enough: `ctypes.WinDLL` respects it,
    but CTranslate2's own internal loader does not -- it uses the classic
    `LoadLibrary` search order, which honours `PATH`, not
    `AddDllDirectory`-registered directories. Prepending to `PATH` is the
    mechanism that actually works, and this must run *before* `faster_whisper`
    (and transitively `ctranslate2`) is imported, in case any CUDA probing
    happens at import time rather than lazily on first inference call.
    """
    if sys.platform != "win32":
        return
    torch_lib = str(Path(torch.__file__).parent / "lib")
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 -- must follow the PATH fix above

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

    def to_json(self) -> str:
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


def transcribe(audio_path: Path, model: WhisperModel, language: str | None = None) -> Transcript:
    """Transcribe `audio_path`, returning a flat, word-level `Transcript`.

    `language=None` auto-detects from the first ~30s of audio. That window can
    be title music or other non-speech content, which is a known trigger for
    Whisper mis-detecting the language and then producing repetition loops or
    hallucinated text (observed on the reference video: auto-detect landed on
    "la" at 46% confidence, and the transcript opened with a phrase repeated
    three times verbatim). Passing an explicit `language` skips detection
    entirely and avoids this class of failure -- worth doing whenever the
    video's language is known ahead of time, at the cost of the generality
    auto-detection provides for an unknown input.

    Words without a usable timestamp (rare, but faster-whisper can emit them
    for e.g. non-speech noise) are dropped rather than kept with a fabricated
    time -- a wrong timestamp is worse than a missing word here, since the
    whole pipeline downstream trusts word.start as ground truth.
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
