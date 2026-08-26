# Sample input and output

A real, verified run against the reference video and dialogue.

**Command:**

```bash
uv run quest1 --url https://ok.ru/video/248244667877 --dialogue "My mind rebels at stagnation" --language en
```

**Terminal output:**

```
Timestamp : 05:25.222
Frame     : 7798
Text      : "My mind rebels at stagnation."
Image     : outputs\answer_frame.png
```

**Files in this directory:**

- [`result.json`](result.json) -- the machine-readable record `--output-dir` writes alongside the image
- [`answer_frame.png`](answer_frame.png) -- the actual extracted frame

The full pipeline (download, transcribe, fuzzy-match, forced-align, extract) ran
against the live video for this; nothing here is fabricated. `--language en` was
added after that auto-detect can mis-fire on a non-speech opening -- see
[APPROACH.md](../APPROACH.md).
