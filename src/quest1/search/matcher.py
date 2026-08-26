"""Fuzzy-match the target dialogue against the ASR transcript.

Matching policy: score decides whether a candidate counts as a real match
(threshold); time decides which real match is first (earliest start among
survivors) -- never score, since a higher score just means ASR transcribed
that occurrence more cleanly, not that it happened first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from ..audio.transcribe import Transcript

#: Empirically tuned, not theoretical -- see DESIGN.md for the false
#: positives/negatives that moved this from 70 to 81. Note the survivor check
#: below is `score >= threshold`, so this value itself is excluded only by
#: scores strictly below it.
DEFAULT_THRESHOLD = 81.0
#: How many words shorter/longer than the target a candidate window may be.
#: ASR wording drifts in length as well as content (contractions, a dropped
#: article), so the window size is searched over a small range instead of
#: being fixed to the target's exact word count.
WINDOW_SLACK = 3


def normalize(text: str) -> str:
    """Casefold and strip punctuation so ASR output and the target compare fairly."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Candidate:
    """One word span in the transcript scored against the target dialogue."""

    word_start: int
    word_end: int  # exclusive
    start: float
    end: float
    text: str
    score: float


def find_candidates(
    transcript: Transcript,
    dialogue: str,
    threshold: float = DEFAULT_THRESHOLD,
    window_slack: int = WINDOW_SLACK,
) -> list[Candidate]:
    """Every genuine occurrence of `dialogue` in `transcript`, earliest first.

    A genuine occurrence is one word span whose normalised text scores >=
    `threshold` against the normalised target, after collapsing overlapping
    detections (multiple window sizes at nearby start positions all firing on
    the same underlying occurrence) down to one best-scoring candidate each.
    """
    target_norm = normalize(dialogue)
    n = len(target_norm.split())
    words = transcript.words
    if n == 0 or not words:
        return []

    norm_words = [normalize(w.text) for w in words]
    min_window = max(1, n - window_slack)
    max_window = n + window_slack

    best_at_start: dict[int, Candidate] = {}
    for i in range(len(words)):
        best_score = -1.0
        best_window = None
        for window in range(min_window, min(max_window, len(words) - i) + 1):
            window_text = " ".join(w for w in norm_words[i : i + window] if w)
            if not window_text:
                continue
            score = fuzz.ratio(target_norm, window_text)
            if score > best_score:
                best_score = score
                best_window = window
        if best_window is not None:
            j = i + best_window
            best_at_start[i] = Candidate(
                word_start=i,
                word_end=j,
                start=words[i].start,
                end=words[j - 1].end,
                text=" ".join(w.text for w in words[i:j]),
                score=best_score,
            )

    # Non-max suppression: keep the best-scoring candidate in each cluster of
    # overlapping/near start positions, so one real occurrence doesn't produce
    # several near-duplicate entries in the result.
    ordered = sorted(best_at_start.values(), key=lambda c: c.score, reverse=True)
    accepted: list[Candidate] = []
    for cand in ordered:
        if any(cand.word_start < a.word_end and a.word_start < cand.word_end for a in accepted):
            continue
        accepted.append(cand)

    survivors = [c for c in accepted if c.score >= threshold]
    survivors.sort(key=lambda c: c.start)
    return survivors


def best_match(
    transcript: Transcript, dialogue: str, threshold: float = DEFAULT_THRESHOLD
) -> Candidate | None:
    """The first genuine occurrence of `dialogue`, or None if it never appears."""
    candidates = find_candidates(transcript, dialogue, threshold)
    return candidates[0] if candidates else None


def best_near_miss(transcript: Transcript, dialogue: str) -> Candidate | None:
    """The single best-scoring candidate regardless of threshold, for
    diagnostics when nothing clears the real threshold: shows what came
    closest instead of a bare "not found"."""
    candidates = find_candidates(transcript, dialogue, threshold=0.0)
    return max(candidates, key=lambda c: c.score) if candidates else None
