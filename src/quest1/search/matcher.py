"""Stage 3 - match the target dialogue against transcript / OCR text.

Planned: normalise (casefold, strip punctuation), then fuzzy-match so ASR and OCR
errors do not cause a miss. Returns candidates with a confidence score.
"""
