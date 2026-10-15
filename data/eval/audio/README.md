Add consented, de-identified WAV recordings here. Put clips and reference transcripts in `manifest.json`:

```json
[
  {"file": "ml_01.wav", "language": "ml", "transcript": "റിപ്പോർട്ടിൽ എന്താണ്?"},
  {"file": "en_01.wav", "language": "en", "transcript": "What is in the report?"}
]
```

Run `python eval/eval_stt.py`. Do not add real patient recordings or identifiers to this repository.
