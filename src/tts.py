"""Swappable TTS interface; gTTS is the default engine."""
from __future__ import annotations

import io
import re
from typing import Protocol

from gtts import gTTS


class TTSEngine(Protocol):
    def synthesize(self, text: str, language: str) -> bytes: ...


def prepare_for_speech(text: str, language: str) -> str:
    """Space digit/unit boundaries; retain source numbers and English medicine names."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"\b(mg|mcg|ml|mmol|g|kg)\b", lambda m: " " + m.group(1) + " ", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip()


class GTTSVoice:
    def synthesize(self, text: str, language: str) -> bytes:
        if language not in {"ml", "en"}:
            raise ValueError("Unsupported speech language")
        buffer = io.BytesIO()
        gTTS(text=prepare_for_speech(text, language), lang=language).write_to_fp(buffer)
        return buffer.getvalue()
