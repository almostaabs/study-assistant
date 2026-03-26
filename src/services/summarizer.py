"""
Summarization service module.
Phase 3: Improved prompt and structured output for exam-ready summaries.
"""

from .ollama import generate_summary


def summarize_text(text: str) -> str:
    """
    Generate an exam-ready summary of the provided text.

    Features:
    - Structured output with title and sections
    - Strict bullet format (12-15 words max)
    - Consistent formatting
    - Clean post-processing

    Args:
        text: The text to summarize.

    Returns:
        Formatted bullet-point summary ready for studying.
    """
    if not text or not text.strip():
        return "Error: No text provided to summarize."

    try:
        return generate_summary(text)
    except ConnectionError as e:
        return (
            f"Error: {e}\n\n"
            "To use this feature:\n"
            "1. Install Ollama from https://ollama.com\n"
            "2. Start Ollama: ollama serve\n"
            "3. Pull a model: ollama pull llama3.2"
        )
    except Exception as e:
        return f"Error generating summary: {e}"
