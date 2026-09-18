"""Tests for the Ollama service: response cleanup and model resolution.

All HTTP calls are mocked -- the suite never needs a running Ollama.
"""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from services import ollama


def fake_response(payload: dict):
    """Build a context-manager stand-in for urllib.request.urlopen()."""
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


# --- _process_response: parsing, dedup, word limit ---


def test_title_becomes_h1():
    out = ollama._process_response("TITLE: Photosynthesis\n- Plants convert light into chemical energy")
    assert out.startswith("# Photosynthesis")


def test_noise_lines_are_dropped():
    raw = (
        "Here is a summary of the text:\n"
        "- Cells divide during mitosis to produce identical daughter cells\n"
        "In conclusion, mitosis matters\n"
    )
    out = ollama._process_response(raw)
    assert out == "- Cells divide during mitosis to produce identical daughter cells"


def test_duplicate_bullets_are_removed_case_insensitively():
    raw = "- Mitosis produces two identical cells\n- MITOSIS PRODUCES TWO IDENTICAL CELLS\n"
    assert out_lines(ollama._process_response(raw)) == ["- Mitosis produces two identical cells"]


def test_bullets_are_truncated_to_fifteen_words():
    long_bullet = "- " + " ".join(f"word{i}" for i in range(25))
    out = ollama._process_response(long_bullet)
    assert out.endswith("...")
    assert len(out.lstrip("- ").rstrip(".").split()) == 15


def test_section_headers_are_normalized():
    raw = "**Key Concepts**\n- Energy flows one way through an ecosystem\n"
    out = ollama._process_response(raw)
    assert "## Key Concepts" in out


def test_short_and_numeric_lines_are_skipped():
    assert ollama._process_response("1.\n- ok\n") == ""


def test_blank_response_yields_empty_string():
    assert ollama._process_response("   \n\n  ") == ""


def out_lines(text):
    return [line for line in text.split("\n") if line.strip()]


# --- generate_summary: HTTP handling ---


def test_generate_summary_posts_prompt_and_cleans_response():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return fake_response({"response": "TITLE: Cells\n- Cells are the basic unit of life"})

    with patch.object(ollama.urllib.request, "urlopen", fake_urlopen):
        out = ollama.generate_summary("some text", model="testmodel")

    assert captured["url"] == ollama.OLLAMA_API_URL
    assert captured["body"]["model"] == "testmodel"
    assert captured["body"]["stream"] is False
    assert "some text" in captured["body"]["prompt"]
    assert out == "# Cells\n- Cells are the basic unit of life"


def test_generate_summary_truncates_very_long_input():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return fake_response({"response": "- ok bullet point here"})

    with patch.object(ollama.urllib.request, "urlopen", fake_urlopen):
        ollama.generate_summary("~" * 20000, model="testmodel")

    # "~" appears nowhere in the prompt template, so every occurrence is input.
    assert captured["body"]["prompt"].count("~") == 8000


def test_unreachable_ollama_raises_connection_error():
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch.object(ollama.urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            ollama.generate_summary("text", model="testmodel")


def test_malformed_json_raises_runtime_error():
    def fake_urlopen(req, timeout=None):
        return io.BytesIO(b"not json at all")

    with patch.object(ollama.urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(RuntimeError):
            ollama.generate_summary("text", model="testmodel")


def test_missing_response_field_yields_empty_summary():
    with patch.object(ollama.urllib.request, "urlopen", lambda req, timeout=None: fake_response({})):
        assert ollama.generate_summary("text", model="testmodel") == ""


# --- resolve_model ---


def patch_installed(names):
    return patch.object(ollama, "list_installed_models", lambda: names)


def test_explicit_model_wins_over_env(monkeypatch):
    monkeypatch.setenv(ollama.MODEL_ENV_VAR, "env-model")
    assert ollama.resolve_model("explicit-model") == "explicit-model"


def test_env_var_wins_over_installed_models(monkeypatch):
    monkeypatch.setenv(ollama.MODEL_ENV_VAR, "env-model")
    with patch_installed(["llama3.2:3b"]):
        assert ollama.resolve_model() == "env-model"


def test_preferred_model_chosen_over_first_installed(monkeypatch):
    monkeypatch.delenv(ollama.MODEL_ENV_VAR, raising=False)
    with patch_installed(["nomic-embed-text:latest", "llama3.2:3b"]):
        assert ollama.resolve_model() == "llama3.2:3b"


def test_preference_order_is_respected(monkeypatch):
    monkeypatch.delenv(ollama.MODEL_ENV_VAR, raising=False)
    with patch_installed(["mistral:7b", "llama3.2:3b"]):
        assert ollama.resolve_model() == "llama3.2:3b"


def test_falls_back_to_first_installed_when_none_preferred(monkeypatch):
    monkeypatch.delenv(ollama.MODEL_ENV_VAR, raising=False)
    with patch_installed(["some-exotic-model:latest"]):
        assert ollama.resolve_model() == "some-exotic-model:latest"


def test_falls_back_to_default_when_nothing_installed(monkeypatch):
    monkeypatch.delenv(ollama.MODEL_ENV_VAR, raising=False)
    with patch_installed([]):
        assert ollama.resolve_model() == ollama.DEFAULT_MODEL


def test_list_installed_models_parses_tags_payload():
    payload = {"models": [{"name": "llama3.2:3b"}, {"name": "mistral:7b"}, {}]}
    with patch.object(ollama.urllib.request, "urlopen", lambda req, timeout=None: fake_response(payload)):
        assert ollama.list_installed_models() == ["llama3.2:3b", "mistral:7b"]


def test_list_installed_models_returns_empty_when_ollama_down():
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("down")

    with patch.object(ollama.urllib.request, "urlopen", fake_urlopen):
        assert ollama.list_installed_models() == []
