"""
CLI module for the AI Study Assistant.
Handles command-line interface and argument parsing.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.summarizer import summarize_text


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="summarize",
        description="AI Study Assistant - Summarize text content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text",
        type=str,
        help="Text content to summarize",
    )
    source.add_argument(
        "--file",
        type=str,
        help="Path to a UTF-8 text file to summarize",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama model to use. Overrides the STUDY_ASSISTANT_MODEL "
            "environment variable and auto-detection."
        ),
    )

    return parser


def read_file(path: str) -> str:
    """
    Read text from a file.

    Raises:
        SystemExit: If the file is missing, unreadable, or not UTF-8 text.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error: Cannot read file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: File '{path}' is not UTF-8 text.", file=sys.stderr)
        sys.exit(1)


def validate_text(text: str) -> None:
    """
    Validate the input text.

    Raises:
        SystemExit: If text is empty or whitespace only.
    """
    if not text.strip():
        print("Error: Text input cannot be empty.", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Gather input
    text = read_file(args.file) if args.file else args.text

    # Validate input
    validate_text(text)

    # Call service layer
    summary = summarize_text(text, model=args.model)

    # Output result
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
