"""Check Claude API connectivity without printing credentials or report data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from config import settings  # noqa: E402


def main() -> None:
    print(f"Configured model: {settings.claude_model}")
    print(f"API key present: {bool(settings.anthropic_api_key)}")
    if not settings.anthropic_api_key:
        return
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        client.messages.create(
            model=settings.claude_model, max_tokens=16,
            messages=[{"role": "user", "content": "Reply OK"}],
        )
        print("API request: successful")
    except anthropic.APIStatusError as exc:
        error = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
        print(f"HTTP status: {exc.status_code}")
        print(f"Error type: {error.get('type', 'unknown')}")
        print(f"Reason: {error.get('message', 'not provided')}")
    except anthropic.APIError as exc:
        print(f"Connection error: {type(exc).__name__}")


if __name__ == "__main__":
    main()
