"""Make `src` importable the same way the `summarize` entry point does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
