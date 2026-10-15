"""Compute English and Malayalam word error rate from user-provided recordings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jiwer import wer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.stt import transcribe_audio  # noqa: E402
from common import update_results  # noqa: E402


def main() -> None:
    manifest_path = ROOT / "data" / "eval" / "audio" / "manifest.json"
    if not manifest_path.exists():
        print("No audio manifest. See data/eval/audio/README.md")
        update_results("STT WER", "Not run: add consented clips and manifest.json")
        return
    from faster_whisper import WhisperModel
    model = WhisperModel(settings.whisper_model, device="auto", compute_type="default")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    for language in ("ml", "en"):
        selected = [row for row in rows if row["language"] == language]
        if not selected:
            continue
        references = []
        hypotheses = []
        for row in selected:
            audio_path = (manifest_path.parent / row["file"]).resolve()
            if audio_path.parent != manifest_path.parent.resolve():
                raise ValueError("Audio path must stay in data/eval/audio")
            references.append(row["transcript"])
            hypotheses.append(transcribe_audio(audio_path.read_bytes(), language, model))
        value = wer(references, hypotheses)
        print(f"{language} WER: {value:.3f} (n={len(selected)})")
        update_results(f"STT WER {language}", f"{value:.3f} (n={len(selected)})")


if __name__ == "__main__":
    main()
