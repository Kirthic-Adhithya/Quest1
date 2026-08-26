# Quest1

Given a video and a line of dialogue, find the exact frame where that line is spoken --
report its timestamp, frame number, the matched text, and a saved image of the frame.

- Reference video: https://ok.ru/video/248244667877
- Reference dialogue: "My mind rebels at stagnation"
- See [APPROACH.md](APPROACH.md) for the design and [samples/](samples/) for a real,
  verified input/output example.

## Requirements

- **Windows.** The project only runs on Windows in practice (see `[tool.uv]` in
  `pyproject.toml`); it has not been tested on Linux/macOS.
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management and running commands.
- **An NVIDIA GPU with CUDA** is strongly recommended -- transcription and alignment fall
  back to CPU automatically if none is found, but a full-length video becomes
  impractically slow. Developed and verified against an 8GB card.
- **ffmpeg** is *not* a separate install -- `imageio-ffmpeg` bundles a working binary
  automatically.
- No API keys or external accounts are needed; everything runs locally.

## Install

```bash
git clone <this-repository-url>
cd Quest1
uv sync
```

`uv sync` installs every dependency listed in `pyproject.toml`, including the CUDA build
of PyTorch/torchaudio from the pinned `pytorch-cu121` index. First install downloads
several GB (PyTorch + CUDA libraries); model weights (Whisper, forced alignment) are
downloaded separately on first actual use, not at install time.

## Run (CLI)

```bash
uv run quest1
```

With no arguments, that runs the reference video and dialogue above, auto-detecting the
transcription language. To run your own:

```bash
uv run quest1 --url https://ok.ru/video/248244667877 --dialogue "My mind rebels at stagnation" --language en
```

`--language` is optional but recommended when the language is known -- auto-detect
samples only the first ~30s of audio and can mis-fire on a non-speech opening (see
APPROACH.md). Output:

```
Timestamp : 05:25.222
Frame     : 7798
Text      : "My mind rebels at stagnation."
Image     : outputs\answer_frame.png
```

A machine-readable `outputs/result.json` is written alongside the image on every run,
including a "not found" run. See [samples/](samples/) for this exact example's real
output files.

### CLI flags

| Flag | Meaning |
|---|---|
| `--url` | media URL to analyse |
| `--dialogue` | the line to search for |
| `--quality` | `mobile` / `lowest` / `low` / `sd` / `hd` / `best` (default `sd`) |
| `--language` | force transcription language by code (e.g. `en`); default auto-detect |
| `--threshold` | minimum fuzzy-match score 0-100 to accept a candidate (default 81) |
| `--model-size` | Whisper model size; default `distil-large-v3` for `--language en`, `large-v3` otherwise |
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

## Run (web app)

```bash
uv run quest1-web
```

Then open http://localhost:8000. Paste a URL (or drop a local video file), type a
dialogue, pick a quality and language, and get back the timestamp, frame number, matched
text, match score, and the frame image -- with a toggle to view the matched frame or play
the source video from that exact moment, and a download button for either.

Jobs run through a single-worker queue: only one video is ever processed at a time,
because this pipeline needs most of an 8GB GPU per run and running more than one
concurrently causes severe slowdown rather than a clean error (see APPROACH.md). A second
submission simply waits its turn.

A locally dropped file skips the download step entirely -- it's probed and transcribed
directly, so `--quality` doesn't apply there. At most one uploaded file is kept on disk
at a time, mirroring the download cache's single-video policy.

## Dependencies

Managed entirely through `pyproject.toml` / `uv.lock`; nothing is installed manually.
The notable ones:

| Package | Purpose |
|---|---|
| `yt-dlp` | downloads the source video |
| `av` (PyAV) | probes video properties and does PTS-accurate frame decoding/seeking |
| `faster-whisper` + `torch` | speech-to-text (CTranslate2-backed Whisper) |
| `torchaudio` + `uroman` | CTC forced alignment for precise onset timing, with non-Latin script support |
| `rapidfuzz` | fuzzy text matching between the target dialogue and the transcript |
| `pillow` | saves the answer frame as a PNG |
| `fastapi` + `uvicorn` + `python-multipart` | the web app and its file-upload endpoint |
| `imageio-ffmpeg` | bundles ffmpeg, used for audio extraction and remuxing |
| `certifi` | CA bundle -- without it the media CDN fails TLS verification |

See [APPROACH.md](APPROACH.md) for *why* each was chosen over alternatives.

## Network notes

`ok.ru` resets connections intermittently, so extraction is retried up to 5 times with
backoff, and transfers are chunked with resume enabled.

## Project layout

```
src/quest1/
  cli.py        argparse entry point
  inputs.py     URL + dialogue validation
  pipeline.py   shared download+transcribe entry points, and the Result type
  ingest/
    downloader.py download the media (yt-dlp), probe its properties (PyAV), disk cache
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
    app.py             FastAPI backend (POST /api/jobs[/upload], GET /api/jobs/{id}, .../image, .../video)
    static/index.html  the browser UI (self-contained, no build step)
APPROACH.md      design, algorithms, assumptions, and trade-offs
prompts.txt      LLM prompts used during development
samples/         a real, verified input/output example (committed, not gitignored)
data/            downloaded media, model weights, caches (gitignored)
outputs/         results: answer_frame.png, result.json (gitignored)
```

## Tests

```bash
uv run pytest -q
```

Fast tests only (no GPU/model calls) -- error paths, data-structure round-trips, and the
matching/frame-indexing policies directly. The real GPU pipeline is verified by running
it against the reference video; see [samples/](samples/) for recorded output.
