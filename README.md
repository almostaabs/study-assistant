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
./summarize --file notes.txt
```

`--text` and `--file` are mutually exclusive; one is required. `--file` expects
UTF-8 text.

### Choosing the model

```bash
./summarize --file notes.txt --model mistral:7b   # explicit
export STUDY_ASSISTANT_MODEL=llama3.1:8b          # for the whole shell session
```

Precedence: `--model`, then `STUDY_ASSISTANT_MODEL`, then the first installed
model matching a known-good list (llama3.2, llama3.1, llama3, mistral, qwen2.5,
phi3, gemma2), then whatever Ollama lists first, then `llama3.2`.

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
tests/                      pytest suite (Ollama HTTP calls are mocked)
```

## Development

```bash
pip install pytest ruff
pytest -q
ruff check .
```

Tests never touch the network, so they run without Ollama installed. CI runs the
same two commands on push and pull request (Python 3.9 and 3.12).

## Known limitations

- Input is truncated at 8000 characters -- long documents are not chunked.
- Only plain text: no PDF, DOCX, or Markdown-aware extraction.
- Model selection is name-based. The preference list is a heuristic, not a
  quality measurement, and an unknown model is used as-is with no warning.
- Output quality depends entirely on the local model; small models sometimes
  ignore the 12-15 word rule (bullets are then truncated with an ellipsis).
- No streaming output -- the CLI blocks until the whole summary is ready.
