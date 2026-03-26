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

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text content to summarize",
    )

    return parser


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

    # Validate input
    validate_text(args.text)

    # Call service layer
    summary = summarize_text(args.text)

    # Output result
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
