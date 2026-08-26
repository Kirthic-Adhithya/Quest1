# Quest1

Find the exact video frame in which a given dialogue first appears, and extract its text.

**Problem:** given a media URL and a dialogue line, report the timestamp, frame number, dialogue text, and the frame image.

- Reference video: https://ok.ru/video/248244667877
- Reference dialogue: "My mind rebels at stagnation"

## Status

Fully implemented: download -> probe -> transcribe -> fuzzy-match -> forced-align ->
extract frame -> report, plus a browser UI over the same pipeline. See
[DESIGN.md](DESIGN.md) for the architecture and design decisions.

## Web app

A browser UI over the same pipeline: paste a URL and a dialogue, pick a
download quality, and get back the timestamp, frame number, extracted text,
match score, and the frame image, plus buttons to download the frame or the
source video.

```bash
uv run quest1-web
```

Then open http://localhost:8000. Jobs run through a single-worker queue -- only one
video is ever processed at a time, because this pipeline needs most of an 8GB GPU per
run and running more than one concurrently causes severe slowdown rather than a clean
error (see DESIGN.md). A second submission simply waits its turn.

## Run

```bash
uv run quest1
```

That uses the reference video and dialogue, auto-detecting the transcription language.
Auto-detect can mis-fire on a non-speech opening (documented in DESIGN.md) -- pass
`--language en` for a cleaner transcript when the language is known:

```bash
uv run quest1 --url https://ok.ru/video/248244667877 --dialogue "My mind rebels at stagnation" --language en
```

Output:

```
Timestamp : 05:25.222
Frame     : 7798
Text      : "my mind rebels at stagnation"
Image     : outputs\answer_frame.png
```

A machine-readable `outputs/result.json` is written alongside the image on every run,
including a "not found" run (see below).

| Flag | Meaning |
|---|---|
| `--url` | media URL to analyse |
| `--dialogue` | the line to search for |
| `--quality` | `mobile` / `lowest` / `low` / `sd` / `hd` / `best` (default `sd`) |
| `--language` | force transcription language (e.g. `en`); default auto-detect |
| `--threshold` | minimum fuzzy-match score 0-100 to accept a candidate (default 81) |
| `--model-size` | Whisper model size (default `large-v3`); smaller = faster dev iteration |
| `--media-dir` | download/transcript cache location (default `data/media`) |
| `--model-dir` | Whisper weights cache (default `data/models`) |
| `--align-model-dir` | forced-alignment weights cache (default `data/models`) |
| `--output-dir` | where the report + frame image are written (default `outputs`) |
| `--open` | open the answer frame image after a successful run |

Every stage caches to disk (download, decoded audio, transcript, model weights), so a
repeat run against the same video/language/quality takes seconds, not minutes.

### If the dialogue isn't found

Exits non-zero and reports the best rejected candidate for diagnosis, rather than
returning a low-confidence guess as if it were the answer:

```
No confident match for "..." (threshold=81).
Best candidate (below threshold, NOT returned as the answer): "..." at 20:26.360, score=55.4
```

### Network notes

`ok.ru` resets connections intermittently, so extraction is retried up to 5 times with
backoff, and transfers are chunked with resume enabled. `certifi` is a required
dependency: without a CA bundle the media CDN fails TLS verification.

## Layout

```
src/quest1/
  cli.py        argparse entry point
  inputs.py     URL + dialogue validation
  pipeline.py   shared download+transcribe entry point, and the Result type
  ingest/
    downloader.py download the media (yt-dlp) and probe its properties (PyAV)
  audio/
    extract.py    video -> 16kHz mono WAV
    transcribe.py word-level speech-to-text via faster-whisper
    align.py      forced alignment via torchaudio MMS_FA, romanised via uroman
                   for non-Latin scripts
  search/
    matcher.py    fuzzy dialogue matching against the transcript
  video/
    frames.py     PTS-accurate frame extraction
  report/
    output.py     final report + JSON record
  web/
    jobs.py            single-worker job queue wrapping the pipeline
    app.py             FastAPI backend (POST /api/jobs, GET /api/jobs/{id}, .../image, .../video)
    static/index.html  the browser UI (self-contained, no build step)
DESIGN.md        architecture, design decisions, and key findings
PROMPTS.txt      LLM prompts used during development
data/            downloaded media, model weights, caches (gitignored)
outputs/         results: answer_frame.png, result.json (gitignored)
```

## Tests

```bash
uv run pytest -q
```

Fast tests only (no GPU/model calls) -- error paths, data-structure round-trips, and the
matching/frame-indexing policies directly. The real GPU pipeline is verified by running
it against the reference video; see DESIGN.md for recorded results.
