"""Tests for summarize_text() -- validation, model pass-through, error messages."""

from unittest.mock import patch

from services import summarizer


def test_empty_input_returns_error_without_calling_ollama():
    with patch.object(summarizer, "generate_summary") as gen:
        assert summarizer.summarize_text("   ") == "Error: No text provided to summarize."
    gen.assert_not_called()


def test_model_is_passed_through():
    with patch.object(summarizer, "generate_summary", return_value="- ok") as gen:
        summarizer.summarize_text("text", model="mistral:7b")
    gen.assert_called_once_with("text", model="mistral:7b")


def test_connection_error_becomes_setup_instructions():
    with patch.object(summarizer, "generate_summary", side_effect=ConnectionError("no ollama")):
        out = summarizer.summarize_text("text")
    assert "no ollama" in out
    assert "ollama serve" in out


def test_other_errors_are_reported_not_raised():
    with patch.object(summarizer, "generate_summary", side_effect=RuntimeError("boom")):
        assert summarizer.summarize_text("text") == "Error generating summary: boom"
