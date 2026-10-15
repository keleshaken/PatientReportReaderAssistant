# Patient Document Voice Assistant

A mobile-friendly Streamlit app that reads an English hospital PDF or photo and automatically presents a short Malayalam explanation. Patients can ask follow-up questions in Malayalam or English by text or voice. Every supported answer shows the original English source line, page and section. The included reports are entirely fictional and are marked as demonstrations.


## Requirements and setup

Python 3.10 or later and [Ollama for Windows](https://docs.ollama.com/windows) are required for the default, no-API-credit setup. Ollama runs the answer model on your computer. The default `qwen3:4b-instruct` download is about 2.5 GB and needs additional RAM and disk space while running. Initial E5 and Whisper model downloads and gTTS need internet access. A working microphone requires browser permission and HTTPS or localhost. Scanned PDFs and photos require the Tesseract executable installed on Windows; `pytesseract` alone is not enough. Text-based PDFs do not require Tesseract.

**Windows Command Prompt, existing installation:**

1. Install Ollama from the official [Windows download](https://ollama.com/download/windows). Close and reopen Command Prompt after installation.
2. Download the local model: `ollama pull qwen3:4b-instruct`.
3. In your existing project folder, run:

```bat
cd /d "C:\Users\krish\OneDrive\Documents\ChatGPT\Gen AI\patient-doc-assistant"
.venv\Scripts\activate.bat
python -m streamlit run app.py
```

The app defaults to `LLM_BACKEND=ollama` even if your existing `.env` contains an Anthropic key. You do not need to delete your key to use local mode. To set it explicitly, add `LLM_BACKEND=ollama` to `.env`. Never share that file or your key.

For a fresh install, create a Python virtual environment, install `requirements.txt`, and copy `.env.example` to `.env` before the steps above. On macOS/Linux use `source .venv/bin/activate`. The first run downloads the E5 and Whisper models. `OLLAMA_MODEL` and `WHISPER_MODEL` can be changed in `.env`. The smaller `qwen3:1.7b` model can be tried on a lower-memory PC by pulling it and changing `OLLAMA_MODEL`, but its Malayalam and JSON quality may be lower.

Optional Anthropic mode still works when credits are available: set `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY` in `.env`. A Claude chat subscription does not itself supply API credits.

Streamlit file watching is disabled in `.streamlit/config.toml` because its Windows watcher can emit a harmless `torch.classes` traceback. Restart `streamlit run app.py` after editing source files.

### Upload speed and progress

Verified summary sections now appear as they finish instead of waiting for all six. The six retrieval queries are encoded in one batch, the embedding model is reused across reports, and Ollama is asked to keep its model loaded for 15 minutes between requests. Audio is generated only when **Listen** is pressed so it does not delay text answers. The interface reports progress and the time taken for each report; citations and grounding checks still run before a section is shown.

Short reports (up to 6,000 characters of chunked source text) go directly to Qwen with their source IDs, avoiding the E5 model download and indexing. All sections still pass citation and grounding validation. Longer reports use E5 retrieval and need its one-time model download. Actual processing time depends on CPU/GPU, report length and OCR; immediate results are not guaranteed. On this installation the selected Ollama model is `qwen3:1.7b` for a smaller download and lower processing load, with potentially weaker Malayalam quality than the 4B default.

## Use

This installation uses `TRANSLATION_BACKEND=google`: extracted report text is sent to Google Translate and displayed page by page in Malayalam alongside the English original. It requires internet but no API key, Qwen, embeddings, or local model download. Translations stay in Streamlit session state, and **Clear my data** clears them. Listen sends the selected text to Google's speech service. This no-key web connector is intended for a demo; it is not the supported Google Cloud Translation API and can be blocked or rate-limited. Numbers are compared with the original and mismatches are flagged, but this does not establish clinical accuracy. Review medicine names, doses and instructions against the source.

Online mode also includes **Ask Qwen**: Google translates Malayalam questions into English, the installed local Qwen model selects a short answer from the English report, and Google translates the answer into Malayalam. Exact sentence quotations or a grounding audit, citation membership and numeric checks are required before a supported answer is shown. Each answer includes the English answer and original source lines. This is document Q&A, not medical advice. Longer documents use bounded lexical retrieval without downloading an embedding model. Qwen can still fail to find or verify an answer. Set `TRANSLATION_BACKEND=local` to use the original six-section explanation and Q&A flow below.

1. Upload a PDF, JPG, PNG, or WEBP. The six Malayalam cards appear automatically.
2. Open **Source** below a card to see the exact English lines and page. The medicines card has a four-column table.
3. Press **Listen** on a card or above all cards to hear it.
4. Choose Malayalam or English. Record a question or type one. After recording, check and edit the transcript in the text box before pressing **Ask**.
5. Read or play the answer and check its source lines. Press **Clear my data** when finished.

## Build stages

1. English text-only RAG: `src/ingest.py`, `chunking.py`, `embeddings.py`, `vectorstore.py`, `retriever.py`, `qa.py`.
2. Malayalam questions and answers plus the automatic upload explanation: `prompts/`, `explain.py`, `app.py`.
3. Voice input and editable transcript: `stt.py`, Streamlit `st.audio_input`.
4. Voice output: `tts.py` and Listen controls.
5. Safety, fallbacks and session cleanup: `safety.py`, prompts and `SessionDocument`.
6. Synthetic documents and evaluations: `scripts/generate_samples.py`, `data/`, `eval/`.
7. Tests and documentation.

To run stage 1 by itself after setup:

```powershell
python scripts/stage1_english.py data/sample_docs/discharge_dengue.pdf "What is the diagnosis?"
```

## Design decisions and assumptions

- RAG ties each response to retrieved document lines and avoids sending the whole report with every question. Retrieval uses a fresh query even for follow-ups; only the last four turns are passed as reference context.
- Headings and page boundaries form first-level chunks. Long sections split near 400 estimated tokens with about 50 tokens of line overlap. This is an approximation because the splitter intentionally avoids another tokenizer download. Exact line IDs such as `p2-l7` survive splitting.
- E5's `passage: ` and `query: ` prefixes are applied separately. Chroma stores one collection in a temporary directory per Streamlit session.
- The scanned sample is a raster PDF. Empty PDF pages are rendered with pypdfium2, then passed to local Tesseract OCR in local mode. Text PDFs use pdfplumber with pypdf fallback. Optional Anthropic mode uses Claude vision first.
- Chroma telemetry is disabled and PostHog is pinned to a version compatible with this Chroma release to avoid noisy telemetry errors.
- Ollama returns JSON; it is parsed, schema checked, tied to retrieved IDs, checked for unsupported numbers, and audited by a second local model call. Invalid or unsupported answers get one retry and then a safe fallback. The auditor is useful but cannot mathematically prove every claim. A local CPU can take several minutes to prepare all six cards.
- Malayalam speech output keeps English medicine names and numeric doses. Before gTTS, `prepare_for_speech` inserts spaces at digit/letter boundaries and around common units. It does not rewrite values or translate medicine names. gTTS may still pronounce mixed-script text imperfectly.
- **Missing information** is shown as `ഈ വിവരം റിപ്പോർട്ടിൽ ഇല്ല` in explanation cards. Q&A uses the exact language-specific fallback in `safety.py`.
- A Streamlit session can end without a reliable disconnect callback. The app attempts cleanup on clear, object disposal and process exit; abandoned indexes are removed on a later app run after 24 hours by default. For strict immediate deletion on browser close, deploy with an external session lifecycle service. Uploaded bytes are not written to a separate file, but the session index temporarily holds document text on local disk.
- Local mode sends explanation and answer requests only to loopback Ollama. E5 and Whisper model downloads and gTTS can require network access; gTTS sends text for synthesis. Optional Anthropic mode sends report content to Anthropic. This demo does not implement authentication or encrypted storage.



## Evaluation

Regenerate the seven synthetic PDFs and the 40 bilingual questions (20 English, 20 Malayalam, eight unanswerable):

```powershell
python scripts/generate_samples.py
python -m pytest -q
python eval/eval_retrieval.py
python eval/eval_answers.py
python eval/eval_stt.py
```

`eval_retrieval.py` calculates hit@1/2/4 and MRR@4 for the annotated section. If the scanned PDF cannot be OCR'd locally, it reports that document as skipped; this is a changed denominator, not a successful retrieval. `eval_answers.py` uses the selected backend and checks exact fallback on unanswerable questions, presence of citations, numeric answer matches, and an LLM faithfulness judgment. The judge rubric accepts an answer only when **every factual claim** is explicitly supported, all names/numbers/units/doses/dates match, and there is no external advice or interpretation. It rejects missing citations or uncertainty. A local model judging itself is a useful smoke check, not independent validation. `eval_stt.py` uses `jiwer` WER on separately supplied, consented English and Malayalam recordings; see `data/eval/audio/README.md`. Scripts update [results.md](eval/results.md). Unrun metrics remain labeled as such.

## Known limitations

- Malayalam Whisper transcription can be inaccurate for dialects, noise, and English medicine names; users should correct the transcript.
- OCR can miss handwriting, faint scans, tables, or rotated photos. The sample scanned PDF is printed text, not handwriting.
- PDF table extraction can lose reading order. The included table tests a common simple layout but not complex merged cells.
- Semantic auditing uses another LLM and may reject valid text or accept a subtle unsupported paraphrase. The exact source lines let the user verify.
- gTTS requires network access and mixed Malayalam/English pronunciation varies.
- Temporary Chroma files may persist until cleanup after an abrupt browser disconnect or process crash.

## Highest-value next improvements

1. Add a document-specific OCR quality check and a manual transcription correction view before indexing.
2. Collect consented Malayalam speech clips and tune or replace STT/TTS for local dialects and mixed medicine names.
3. Add authenticated, encrypted server-side session management with deterministic deletion and independent clinical review of the groundedness rubric.
