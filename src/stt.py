"""Explicit-language Whisper speech-to-text."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def transcribe_audio(audio_bytes: bytes, language: str, model: Any) -> str:
    """Decode Streamlit audio bytes using the selected language, never detection."""
    if language not in {"ml", "en"}:
        raise ValueError("Choose Malayalam or English")
    if not audio_bytes:
        raise ValueError("No audio was recorded")
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio_bytes)
            path = Path(handle.name)
        segments, _ = model.transcribe(str(path), language=language, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
