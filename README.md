# Quest1

Find the exact video frame in which a given dialogue first appears, and extract its text.

**Problem:** given a media URL and a dialogue line, report the timestamp, frame number, dialogue text, and the frame image.

- Reference video: https://ok.ru/video/248244667877
- Reference dialogue: "My mind rebels at stagnation"

## Status

**All 6 stages are implemented**: download -> probe -> transcribe -> fuzzy-match ->
forced-align -> extract frame -> report. See [DESIGN.md](DESIGN.md) for the full
architecture, and the findings section for real bugs caught while building this (an ok.ru
quality-selection bug, a Whisper language-misdetection failure mode, a `round()` vs
`floor()` off-by-one in frame indexing, and more).

## Run

```bash
uv run quest1
```

That uses the reference video and dialogue, auto-detecting the transcription language. On
the reference video, auto-detect can mis-fire on the non-speech opening (documented in
DESIGN.md) -- pass `--language en` for a cleaner transcript when the language is known:

```bash
uv run quest1 --url https://ok.ru/video/248244667877 --dialogue "My mind rebels at stagnation" --language en
```

Output:

```
Timestamp : 05:25.222
Frame     : 7798
Text      : "my mind rebels its stagnation"
Image     : outputs\answer_frame.png
```

("its" instead of "at" is Whisper's transcription, not the target string -- reported
verbatim as what was actually heard. The fuzzy matcher found it anyway; see DESIGN.md.)

A machine-readable `outputs/result.json` is written alongside the image on every run,
including a "not found" run (see below).

| Flag | Meaning |
|---|---|
| `--url` | media URL to analyse |
| `--dialogue` | the line to search for |
| `--quality` | `mobile` / `lowest` / `low` / `sd` / `hd` / `best` (default `sd`) |
| `--language` | force transcription language (e.g. `en`); default auto-detect |
| `--threshold` | minimum fuzzy-match score 0-100 to accept a candidate (default 70) |
| `--model-size` | Whisper model size (default `large-v3`); smaller = faster dev iteration |
| `--media-dir` | download/transcript cache location (default `data/media`) |
| `--model-dir` | Whisper weights cache (default `data/models`) |
| `--align-model-dir` | forced-alignment weights cache (default `data/models`) |
| `--output-dir` | where the report + frame image are written (default `outputs`) |
| `--open` | open the answer frame image after a successful run |

Every stage caches to disk (download, decoded audio, transcript, model weights), so a
repeat run against the same video/language/quality is seconds, not minutes.

### If the dialogue isn't found

Exits non-zero and reports the best rejected candidate for diagnosis, rather than
returning a low-confidence guess as if it were the answer:

```
No confident match for "..." (threshold=70).
Best candidate (below threshold, NOT returned as the answer): "..." at 20:26.360, score=55.4
```

### Network notes

`ok.ru` resets connections intermittently, so extraction is retried up to 5 times with
backoff, and transfers are chunked with resume enabled. `certifi` is a required
dependency: without a CA bundle the media CDN fails TLS verification.

## Layout

```
src/quest1/
  cli.py        argparse entry point, wires all 6 stages together
  inputs.py     URL + dialogue validation
  pipeline.py   library-level stage orchestrator (run_transcription/run_match/run_pipeline)
  ingest/
    downloader.py download + probe the media (stage 1)
  audio/
    extract.py    video -> 16kHz mono WAV (stage 2 input)
    transcribe.py word-level ASR via faster-whisper (stage 2)
    align.py      forced alignment via torchaudio MMS_FA, romanised via uroman
                   for non-Latin scripts (stage 4)
  search/
    matcher.py    fuzzy matching + threshold/earliest-wins policy (stage 3)
  video/
    frames.py     PTS-accurate frame extraction (stage 5)
  report/
    output.py     final report + JSON record (stage 6)
DESIGN.md        design, approach, and every real bug found while building this
PROMPTS.txt      LLM prompts used
data/            downloaded media, model weights, caches (gitignored)
outputs/         results: answer_frame.png, result.json (gitignored)
```

Every file under `src/quest1/` is imported by the pipeline or the CLI -- nothing here is
a stub. (Two early stub files, `utils/text.py` and `video/ocr.py`, were removed once OCR
was ruled out as unnecessary -- see the "on-screen dialogue" ambiguity discussion in
DESIGN.md -- and `matcher.normalize()` covers what `utils/text.py` was meant to hold.)

## Tests

```bash
uv run pytest -q
```

44 tests, all fast (no GPU/model calls) -- they cover error paths, data-structure
round-trips, and the matching/frame-indexing policies directly, with the real GPU
pipeline verified manually against the reference video (see DESIGN.md for the actual
recorded runs and their outputs).
