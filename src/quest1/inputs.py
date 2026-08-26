"""Parsing and validation of the two program inputs: a media URL and a target dialogue.

Kept deliberately separate from the CLI so the same validation is reusable from
tests, a notebook, or a future web/API entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")


class InvalidInputError(ValueError):
    """Raised when a URL or dialogue string cannot be used by the pipeline."""


@dataclass(frozen=True)
class Job:
    """One unit of work: find `dialogue` inside the media at `url`."""

    url: str
    dialogue: str

    @property
    def host(self) -> str:
        """Domain of the media URL, e.g. "ok.ru"."""
        return urlparse(self.url).netloc


def parse_url(raw: str) -> str:
    """Validate a media URL and return it normalised (whitespace stripped).

    We only check that it is a well-formed absolute http(s) URL. Deciding whether
    the host is actually downloadable is the downloader's job, not ours -- the
    problem statement says the evaluator may swap in a different video.
    """
    url = raw.strip()
    if not url:
        raise InvalidInputError("URL is empty.")

    parts = urlparse(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise InvalidInputError(
            f"URL must start with http:// or https:// (got {parts.scheme or 'no scheme'!r})."
        )
    if not parts.netloc:
        raise InvalidInputError(f"URL has no host: {url!r}")
    return url


def parse_dialogue(raw: str) -> str:
    """Validate the dialogue we are searching for and collapse stray whitespace."""
    dialogue = " ".join(raw.split())
    if not dialogue:
        raise InvalidInputError("Dialogue text is empty.")
    return dialogue


def build_job(url: str, dialogue: str) -> Job:
    """Validate both inputs and build the Job the rest of the pipeline runs on."""
    return Job(url=parse_url(url), dialogue=parse_dialogue(dialogue))
