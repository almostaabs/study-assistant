"""
Ollama integration service module.
Handles communication with local Ollama API.
"""

import json
import os
import urllib.request
import urllib.error
import re
from typing import Optional


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Prompt template for exam-ready summaries
SUMMARY_PROMPT_TEMPLATE = """You are a study assistant that creates exam-ready summaries.

STRICT RULES:
- Output ONLY bullet points
- Each bullet: 12-15 words maximum
- One idea per bullet only
- No paragraphs, no introductions, no conclusions
- No explanatory text before or after bullets

OUTPUT FORMAT:
TITLE: <extract or infer a concise title>

- First key point here (12-15 words max)
- Second key point here (12-15 words max)
- Third key point here (12-15 words max)

If multiple topics exist, group with headers:
## Section Name
- Bullet point (12-15 words max)
- Bullet point (12-15 words max)

SUMMARIZE THIS TEXT:
{text}
"""


# Models known to do well at short structured summaries, best first.
# Matched as a prefix against installed model names (so "llama3.2:3b" matches).
PREFERRED_MODELS = ("llama3.2", "llama3.1", "llama3", "mistral", "qwen2.5", "phi3", "gemma2")

DEFAULT_MODEL = "llama3.2"

MODEL_ENV_VAR = "STUDY_ASSISTANT_MODEL"


def list_installed_models() -> list:
    """
    Return the names of models installed in the local Ollama instance.

    Returns an empty list if Ollama is unreachable or reports nothing.
    """
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    return [m["name"] for m in data.get("models", []) if m.get("name")]


def resolve_model(model: Optional[str] = None) -> str:
    """
    Decide which Ollama model to use.

    Precedence:
    1. Explicit argument (``--model``).
    2. The STUDY_ASSISTANT_MODEL environment variable.
    3. The first installed model matching PREFERRED_MODELS.
    4. The first installed model, whatever it is.
    5. DEFAULT_MODEL.

    Args:
        model: Explicitly requested model name, if any.

    Returns:
        Model name string.
    """
    if model:
        return model

    env_model = os.environ.get(MODEL_ENV_VAR, "").strip()
    if env_model:
        return env_model

    installed = list_installed_models()
    for preferred in PREFERRED_MODELS:
        for name in installed:
            if name.startswith(preferred):
                return name

    return installed[0] if installed else DEFAULT_MODEL


def generate_summary(text: str, model: Optional[str] = None) -> str:
    """
    Generate a summary using Ollama LLM.

    Args:
        text: The text to summarize.
        model: Optional model name. If not provided, resolve_model() decides.

    Returns:
        Formatted, structured bullet-point summary.

    Raises:
        ConnectionError: If Ollama is not running or unreachable.
        RuntimeError: If the API request fails.
    """
    model = resolve_model(model)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(text=text[:8000])  # Limit input size

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for consistency
            "num_predict": 2048
        }
    }

    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return _process_response(data.get("response", ""))

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_API_URL}. "
            f"Make sure Ollama is running. Error: {e}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid response from Ollama: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def _process_response(response: str) -> str:
    """
    Process and clean the LLM response into structured output.

    Args:
        response: Raw response from Ollama.

    Returns:
        Clean, structured summary with title and sections.
    """
    lines = response.strip().split("\n")
    output_lines = []
    seen_bullets = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip common prefixes/noise
        if _is_noise_line(line):
            continue

        # Handle TITLE line
        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
            if title:
                output_lines.append(f"# {title}")
            continue

        # Handle section headers
        if line.startswith("##") or line.startswith("**"):
            section = line.strip("#* ")
            if section:
                output_lines.append(f"\n## {section}")
            continue

        # Handle bullet points
        bullet = _normalize_bullet(line)
        if bullet:
            # Deduplicate
            bullet_key = bullet.lower().strip("- ")
            if bullet_key and bullet_key not in seen_bullets:
                seen_bullets.add(bullet_key)
                # Enforce word limit
                bullet = _enforce_word_limit(bullet, max_words=15)
                output_lines.append(bullet)

    result = "\n".join(output_lines)
    return _final_clean(result)


def _is_noise_line(line: str) -> bool:
    """Check if line is noise/intro/conclusion to skip."""
    noise_prefixes = [
        "here is", "here are", "summary:", "key points:",
        "bullet points:", "the text discusses", "this text",
        "in conclusion", "to summarize", "output:", "note:",
        "as requested", "following is", "below are"
    ]
    line_lower = line.lower()
    return any(line_lower.startswith(p) for p in noise_prefixes)


def _normalize_bullet(line: str) -> Optional[str]:
    """Normalize a line to a standard bullet format."""
    # Remove existing bullet markers
    clean = line.lstrip("-*•").strip()
    if not clean or len(clean) < 5:
        return None

    # Skip if it's just a number (like "1.")
    if re.match(r"^\d+\.$", clean):
        return None

    return f"- {clean}"


def _enforce_word_limit(line: str, max_words: int) -> str:
    """Enforce maximum words per bullet."""
    # Extract the bullet content
    content = line.lstrip("- ").strip()
    words = content.split()

    if len(words) <= max_words:
        return line

    # Truncate and add ellipsis
    truncated = " ".join(words[:max_words])
    return f"- {truncated}..."


def _final_clean(text: str) -> str:
    """Final cleanup of the output."""
    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Ensure no trailing whitespace
    lines = [r.rstrip() for r in text.split("\n")]
    return "\n".join(lines).strip()
