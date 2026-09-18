# study-assistant

A small CLI that turns pasted text into an exam-ready, bulleted summary using a
local Ollama model. Nothing leaves your machine.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
  ```bash
  ollama serve
  ollama pull llama3.2
  ```

## Usage

```bash
./summarize --text "paste the text you want summarized here"
```

The CLI sends the text to your local Ollama instance with a prompt that
enforces short bullets (12-15 words each), strips filler like "Here is a
summary of...", deduplicates repeated points, and groups output under `##`
section headers when the source text covers multiple topics.

If Ollama isn't running, it fails with instructions instead of a stack trace.

## How it's structured

```
summarize                  entry point
src/cli/main.py             argument parsing, validation
src/services/summarizer.py  summarize_text() -- the public function the CLI calls
src/services/ollama.py      talks to the Ollama HTTP API, cleans up the response
```

## Known limitations

- Single command (`--text`); no file input yet.
- No automated tests.
- Model auto-detection just picks whatever Ollama reports as the first
  installed model -- it doesn't check that it's actually good at this task.
