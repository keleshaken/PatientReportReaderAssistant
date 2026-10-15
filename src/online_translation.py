"""Session-owned Google web translation for the no-key demonstration mode.

This is a web connector, not Google's supported Cloud Translation API.
"""
from collections import Counter
from dataclasses import dataclass
import re

import httpx


class OnlineTranslationError(RuntimeError):
    pass


@dataclass
class TranslatedPage:
    page: int
    original: str
    translated: str
    numbers_match: bool


def split_translation_text(text: str, limit: int = 2400) -> list[str]:
    """Bound requests while retaining all source characters, including long lines."""
    parts = []
    while text:
        end = min(limit, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", 0, end), text.rfind(" ", 0, end))
            if boundary > end // 2:
                end = boundary + 1
        parts.append(text[:end])
        text = text[end:]
    return parts


class GoogleWebTranslator:
    def __init__(self, transport=None):
        self.client = httpx.Client(timeout=httpx.Timeout(30, connect=10), transport=transport)

    def close(self):
        self.client.close()

    def translate_page(self, page: int, text: str, source: str = "en", target: str = "ml") -> TranslatedPage:
        translated = []
        for part in split_translation_text(text):
            if not part.strip():
                translated.append(part)
                continue
            try:
                response = self.client.get("https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": part})
                response.raise_for_status()
                data = response.json()
                value = "".join(segment[0] for segment in data[0] if segment[0])
                if not value.strip():
                    raise ValueError("Empty translation")
                translated.append(value)
            except (httpx.HTTPError, ValueError, TypeError, IndexError, KeyError) as exc:
                raise OnlineTranslationError(
                    "Google translation is unavailable or rate-limited. Please retry shortly. "
                    "Your original report is unchanged.") from exc
        result = "\n".join(translated)
        numbers = lambda value: Counter(re.findall(r"\d+(?:[.,:/-]\d+)*", value))
        return TranslatedPage(page, text, result, numbers(text) == numbers(result))
