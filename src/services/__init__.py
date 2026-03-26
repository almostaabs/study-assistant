"""
Services module for AI Study Assistant.
"""

from .ollama import generate_summary
from .summarizer import summarize_text

__all__ = ["generate_summary", "summarize_text"]
