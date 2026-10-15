"""Mobile-first Streamlit interface for patient document explanation and Q&A."""
from __future__ import annotations

import atexit
import hashlib
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import httpx

from config import settings
from src.chunking import SourceLine, split_pages
from src.embeddings import E5Embedder
from src.direct_retriever import DirectReportRetriever, MAX_DIRECT_CHARACTERS
from src.explain import ExplanationSection, iter_explanation
from src.ingest import IngestError, extract_document
from src.llm import ClaudeBillingError, LocalModelError, OllamaClient, TextClient, make_client
from src.hybrid_qa import answer_from_report
from src.qa import Answer, answer_question
from src.retriever import Retriever
from src.safety import DISCLAIMER
from src.stt import transcribe_audio
from src.tts import GTTSVoice
from src.vectorstore import VectorStore
from src.online_translation import GoogleWebTranslator, OnlineTranslationError


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# Translation request URLs contain report text. Do not log HTTP request URLs.
logging.getLogger("httpx").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)
ROOT = Path(tempfile.gettempdir()) / "patient-doc-assistant"
ROOT.mkdir(exist_ok=True)


def _safe_delete(path: Path) -> None:
    if path.resolve().parent == ROOT.resolve() and path.name.startswith("session-"):
        shutil.rmtree(path, ignore_errors=True)


def _clear_stale() -> None:
    cutoff = time.time() - settings.session_ttl_hours * 3600
    for child in ROOT.glob("session-*"):
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                _safe_delete(child)
        except OSError:
            pass


_clear_stale()


class SessionDocument:
    """Owns all document data and its temporary, session-specific index."""

    def __init__(self, digest: str, embedder: E5Embedder | None, client: TextClient, chunks=None) -> None:
        self.digest = digest
        self.path = ROOT / f"session-{uuid.uuid4().hex}"
        if embedder is None:
            self.store = None
            self.retriever = DirectReportRetriever(chunks or [])
        else:
            self.store = VectorStore(self.path)
            self.retriever = Retriever(embedder, self.store)
        self.client = client
        self.lines: dict[str, SourceLine] = {}
        self.explanation: list[ExplanationSection] = []
        self.history: list[dict[str, str]] = []
        self.audio_cache: dict[str, bytes] = {}
        atexit.register(_safe_delete, self.path)

    def close(self) -> None:
        self.store = None  # type: ignore[assignment]
        self.retriever = None  # type: ignore[assignment]
        _safe_delete(self.path)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@st.cache_resource(show_spinner="Loading multilingual language model…")
def get_embedder() -> E5Embedder:
    return E5Embedder(settings.embedding_model)


@st.cache_resource(show_spinner="Loading speech recognition model…")
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(settings.whisper_model, device="auto", compute_type="default")


@st.fragment(run_every="15s")
def show_model_status() -> None:
    if settings.translation_backend == "google":
        st.info("Online Malayalam translation · Report text is sent to Google Translate. No local AI model or API key is required.")
        return
    if settings.llm_backend != "ollama":
        return
    try:
        response = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags",
                             timeout=2, trust_env=False)
        response.raise_for_status()
        models = response.json().get("models", [])
        ready = any(item.get("name") == settings.ollama_model or
                    item.get("model") == settings.ollama_model for item in models)
        if ready:
            st.caption("● Local AI ready · Your report stays on this computer during explanation.")
        else:
            st.info("One-time AI model setup is still in progress. PDF explanations will work after the model download finishes. This status refreshes automatically.")
    except (httpx.HTTPError, ValueError):
        st.warning("Open Ollama from the Start menu to enable report explanations.")


def source_display(lines: list[SourceLine], *, expanded: bool = False) -> None:
    if not lines:
        return
    with st.expander("Source · ഉറവിടം", expanded=expanded):
        for line in lines:
            st.markdown(f"**Page {line.page} · {line.section}**  \n> {line.text}")


def speak(text: str, language: str, key: str, document: SessionDocument) -> None:
    if st.button("🔊 Listen · കേൾക്കുക", key=key, disabled=not text.strip()):
        try:
            document.audio_cache[key] = GTTSVoice().synthesize(text, language)
        except Exception:
            LOGGER.warning("TTS failed", exc_info=False)
            st.error("ശബ്ദം തയ്യാറാക്കാനായില്ല. വീണ്ടും ശ്രമിക്കുക. / Audio unavailable. Please retry.")
    if key in document.audio_cache:
        st.audio(document.audio_cache[key], format="audio/mp3")


def show_explanation(document: SessionDocument, *, interactive: bool = True) -> None:
    st.header("Your report, explained · റിപ്പോർട്ട് ലളിതമായി")
    st.caption("Check the English source beneath each section. Listen whenever you need to.")
    if interactive:
        whole = " ".join(f"{section.title}. {section.explanation}" for section in document.explanation)
        speak(whole, "ml", "listen-all", document)
    labels = {"diagnosis": "01 / DIAGNOSIS", "medicines": "02 / MEDICINES",
              "tests": "03 / TEST RESULTS", "follow_up": "04 / FOLLOW-UP",
              "warnings": "05 / WARNING SIGNS", "diet_rest": "06 / DIET & REST"}
    for section in document.explanation:
        with st.container(border=True):
            st.caption(labels.get(section.key, "REPORT SECTION"))
            st.subheader(section.title)
            if section.status == "unverified":
                st.warning(section.explanation)
            else:
                st.write(section.explanation)
            if section.key == "medicines" and section.medicine_rows:
                st.table([{"Drug name": row["drug"], "Dose": row["dose"],
                           "Timing": row["timing"], "Duration": row["duration"]}
                          for row in section.medicine_rows])
                for source in section.sources:
                    st.caption(f"Exact English line · Page {source.page}: {source.text}")
            source_display(section.sources, expanded=section.status == "unverified")
            if section.found and interactive:
                speak(section.explanation + " " + " ".join(" ".join(row.values()) for row in section.medicine_rows),
                      "ml", f"listen-{section.key}", document)


def show_answer(answer: Answer, document: SessionDocument, number: int) -> None:
    with st.container(border=True):
        st.write(answer.text)
        for source in answer.sources:
            st.caption(f"Source · Page {source.page} · {source.section}: {source.text}")
        st.caption(f"Confidence · ഉറപ്പ്: {answer.confidence}")
        audio_key = f"answer-{number}"
        if audio_key in document.audio_cache:
            st.audio(document.audio_cache[audio_key], format="audio/mp3",
                     autoplay=st.session_state.pop("autoplay_answer", False))
        else:
            speak(answer.text, answer.language, audio_key, document)


st.set_page_config(page_title="Patient Document Voice Assistant", page_icon="🩺", layout="wide")
st.markdown("""<style>
.stApp {background: #F4F5F0;}
[data-testid="stHeader"] {display: none;}
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 4rem;}
h1 {letter-spacing: -.045em; font-weight: 750; line-height: 1.12;}
h2 {font-size: 1.65rem; letter-spacing: -.035em;}
h3 {font-size: 1.25rem; letter-spacing: -.02em;}
p, li {line-height: 1.75;}
.stButton button, .stDownloadButton button {min-height: 2.8rem; border-radius: 10px; font-weight: 600;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 20px; border-color: #DFE5DE;}
[data-testid="stFileUploader"] {padding: .4rem 0;}
[data-testid="stFileUploaderDropzone"] {background: #F0F6F1; border: 2px dashed #9BB9A7; border-radius: 16px; padding: 2.1rem 1.2rem;}
[data-testid="stTextArea"] textarea {background: white; font-size: 1rem;}
.brandbar {display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; gap: 12px;}
.brand {display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 700; color: #183D33;}
.brandmark {display: grid; place-items: center; width: 42px; height: 42px; background: #1D5544; color: #E4F3CB; border-radius: 13px; font-size: 28px;}
.brand small {display: block; font-size: 10px; letter-spacing: .16em; color: #73877C; margin-top: 2px;}
.navtag {font-size: 12px; color: #526B60; padding: 8px 14px; border: 1px solid #CCD9CF; border-radius: 30px;}
.st-key-hero {background: #163E32; border-radius: 26px; padding: 34px 40px; margin-bottom: 18px; overflow: hidden;}
.st-key-hero h1 {color: #FAFFF5; font-size: clamp(2.3rem, 4vw, 3.5rem); padding-top: 4px;}
.st-key-hero p {color: #CBDCD0; font-size: 15px; max-width: 470px;}
.eyebrow {font-size: 11px; letter-spacing: .14em; color: #D7EDA6; font-weight: 700; margin-bottom: 12px;}
.hero-pills {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px;}
.hero-pills span {color: #E2EFDB; border: 1px solid #53715F; font-size: 11px; border-radius: 25px; padding: 6px 11px;}
.report-art {position: relative; max-width: 285px; margin: 12px auto 16px; padding: 22px; background: #FFFFF8; border-radius: 14px; transform: rotate(4deg); box-shadow: 15px 18px 0 #295443; color: #233F34;}
.paper-heading {font-size: 11px; letter-spacing: .12em; color: #667D70; border-bottom: 1px solid #DDE4D7; padding-bottom: 12px;}
.paper-title {font-size: 20px; font-weight: 700; margin: 14px 0 10px;}
.paper-line {height: 7px; background: #E5EBDD; border-radius: 8px; margin: 10px 0;}
.paper-answer {background: #E8F0D9; border-left: 3px solid #72924E; padding: 12px; margin-top: 20px; font-size: 13px; line-height: 1.8;}
.paper-seal {position: absolute; right: -17px; bottom: -12px; background: #D9EDA8; color: #24462D; border: 5px solid #163E32; border-radius: 40px; padding: 12px 16px; font-size: 11px; font-weight: 700; transform: rotate(-4deg);}
.step {display: flex; gap: 13px; margin: 19px 0;}
.step b {flex-shrink: 0; width: 29px; height: 29px; background: #E9F0E3; color: #355A43; border-radius: 50%; text-align: center; line-height: 29px; font-size: 12px;}
.step strong {display: block; font-size: 14px; color: #243E32;}
.step span {font-size: 12px; line-height: 1.65; color: #6C7F72;}
.feature-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;}
.feature {padding: 20px; border-top: 2px solid #B7C9AE;}
.feature strong {display: block; color: #264534; font-size: 14px; margin: 8px 0;}
.feature span {color: #708074; font-size: 12px; line-height: 1.7;}
.feature em {font-style: normal; font-size: 21px; color: #6B8A52;}
.st-key-upload-panel, .st-key-guide-panel {padding: 9px; background: white;}
@media(max-width: 700px) {.block-container {padding: 1rem 1rem 3rem;} .st-key-hero {padding: 25px 22px;} .report-art {max-width: 230px; margin: 12px auto 24px;} .feature-row {grid-template-columns: 1fr;} .feature {padding: 12px;} .navtag {font-size: 10px;} h1 {font-size: 2.3rem;}}
</style>""", unsafe_allow_html=True)
st.markdown('''<div class="brandbar"><div class="brand"><span class="brandmark">+</span><div>എന്റെ റിപ്പോർട്ട്<small>PATIENT REPORT COMPANION</small></div></div><span class="navtag">English → മലയാളം</span></div>''', unsafe_allow_html=True)
with st.container(key="hero"):
    intro, illustration = st.columns([1.5, 1], gap="large")
    with intro:
        st.markdown('<div class="eyebrow">A LITTLE CLARITY. A LITTLE MORE CONFIDENCE.</div>', unsafe_allow_html=True)
        st.title("Understand your report.\nഎന്റെ റിപ്പോർട്ട്")
        st.write("Turn your English hospital report into Malayalam. Read the translated content alongside the original, at your own pace."
                 if settings.translation_backend == "google" else
                 "Medical words can feel overwhelming. Turn your English hospital report into a simple Malayalam explanation, one section at a time.")
        st.markdown('<div class="hero-pills"><span>✦ Malayalam explanations</span><span>↗ Original sources</span><span>♫ Listen aloud</span></div>', unsafe_allow_html=True)
    with illustration:
        st.markdown('''<div class="report-art" aria-label="Illustration of a report explained in Malayalam"><div class="paper-heading">YOUR REPORT, MADE CLEAR</div><div class="paper-title">From words to understanding.</div><div class="paper-line"></div><div class="paper-line" style="width:80%"></div><div class="paper-line" style="width:60%"></div><div class="paper-answer">നിങ്ങളുടെ റിപ്പോർട്ട്<br><strong>ലളിതമായ മലയാളത്തിൽ.</strong></div><div class="paper-seal">ENGLISH → മലയാളം</div></div>''', unsafe_allow_html=True)
show_model_status()

if st.button("Clear my data · എന്റെ വിവരങ്ങൾ മായ്ക്കുക"):
    next_uploader_version = st.session_state.get("uploader_version", 0) + 1
    old = st.session_state.get("document")
    if old:
        old.close()
    for key in ("document", "online_document", "hybrid_question", "question_text", "last_audio_hash", "last_answer",
                "answer_number", "processing_seconds", "autoplay_answer"):
        st.session_state.pop(key, None)
    st.success("വിവരങ്ങൾ മായ്ച്ചു. / Your data was cleared.")
    st.session_state["uploader_version"] = next_uploader_version
    st.rerun()

upload_column, guide_column = st.columns([1.6, 1], gap="large")
with upload_column:
    with st.container(border=True, key="upload-panel"):
        st.caption("01 / START HERE")
        st.subheader("A clearer report starts here.")
        st.write("Add your hospital report. We’ll help you read it.")
        uploaded = st.file_uploader("Upload an English hospital report · റിപ്പോർട്ട് ചേർക്കുക",
                                    type=["pdf", "png", "jpg", "jpeg", "webp"],
                                    key=f"upload-{st.session_state.get('uploader_version', 0)}")
        st.caption(f"PDF, JPG, PNG or WEBP · Up to {settings.max_upload_mb} MB")
        sample = Path(__file__).parent / "data/sample_docs/discharge_dengue.pdf"
        if sample.exists():
            st.download_button("↓ Try a sample report", sample.read_bytes(),
                               file_name="sample_discharge_report.pdf", mime="application/pdf")
        st.caption("The sample contains fictional patient information.")
with guide_column:
    with st.container(border=True, key="guide-panel"):
        st.caption("A SIMPLE WAY THROUGH")
        st.subheader("Three small steps.")
        if settings.translation_backend == "google":
            st.markdown('''<div class="step"><b>1</b><div><strong>Add your report</strong><span>Upload an English PDF or a clear photo. Its text is sent to Google Translate.</span></div></div><div class="step"><b>2</b><div><strong>Read the actual content</strong><span>Compare each translated page with its English original.</span></div></div><div class="step"><b>3</b><div><strong>Read. Download. Listen.</strong><span>Save the Malayalam text or listen to a translated page.</span></div></div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="step"><b>1</b><div><strong>Add your report</strong><span>Upload an English PDF or a clear photo.</span></div></div><div class="step"><b>2</b><div><strong>Read it in Malayalam</strong><span>Explore checked sections and their original English sources.</span></div></div><div class="step"><b>3</b><div><strong>Ask. Read. Listen.</strong><span>Type or speak your question, then listen when you need to.</span></div></div>''', unsafe_allow_html=True)
st.caption(DISCLAIMER)
with st.expander("About your data & audio"):
    st.write("In online mode, extracted report text is sent to Google Translate. In local Ollama mode, explanations stay on this computer. Listen sends the selected text to Google's speech service. Translation can contain errors; compare medicines and doses with the original. Use Clear my data when finished.")
if uploaded is None:
    st.markdown('''<div class="feature-row"><div class="feature"><em>▤</em><strong>Six sections. Less confusion.</strong><span>Diagnosis, medicines, tests, follow-up, warning signs, and diet & rest.</span></div><div class="feature"><em>↗</em><strong>See where it came from.</strong><span>Every supported explanation includes the original English source.</span></div><div class="feature"><em>♫</em><strong>Read it. Or hear it.</strong><span>Simple Malayalam text with optional audio, at your own pace.</span></div></div>''', unsafe_allow_html=True)
if uploaded is not None and settings.translation_backend == "google":
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        st.error("The report exceeds the upload size limit.")
        st.stop()
    cached = st.session_state.get("online_document")
    if cached is None or cached.digest != digest:
        st.session_state.pop("online_document", None)
        st.session_state.pop("hybrid_question", None)
        translator = GoogleWebTranslator()
        try:
            started = time.perf_counter()
            with st.status("Translating your report into Malayalam…", expanded=True) as online_status:
                pages = extract_document(data, uploaded.name)
                output = []
                bar = st.progress(0, text="Reading the original report")
                for index, page in enumerate(pages):
                    if page.text.strip():
                        output.append(translator.translate_page(page.page, page.text))
                    bar.progress((index + 1) / len(pages), text=f"Translated page {index + 1} of {len(pages)}")
                cached = SimpleNamespace(digest=digest, pages=output, audio_cache={}, history=[],
                                         unread_pages=[page.page for page in pages if not page.text.strip()],
                                         seconds=time.perf_counter() - started)
                st.session_state["online_document"] = cached
                online_status.update(label="Malayalam translation ready", state="complete", expanded=False)
        except (OnlineTranslationError, IngestError) as exc:
            st.error(str(exc))
            st.button("Retry translation")
            st.stop()
        finally:
            translator.close()
    st.header("Your report in Malayalam · മലയാളം വിവർത്തനം")
    st.caption(f"{len(cached.pages)} page(s) · Translated in {cached.seconds:.1f}s · Google Translate")
    if cached.unread_pages:
        st.warning("Some pages could not be read and were not translated: " + ", ".join(map(str, cached.unread_pages)))
    st.warning("Automatic translation can contain mistakes. Check medicine names, doses and instructions against the English original. This is a translation, not medical advice.")
    st.download_button("Download Malayalam text", "\n\n".join(
        f"Page {page.page}\n{page.translated}" for page in cached.pages),
        file_name="report_malayalam.txt", mime="text/plain")
    for page in cached.pages:
        with st.container(border=True):
            st.subheader(f"Page {page.page}")
            if not page.numbers_match:
                st.error("Numbers differ between this translation and the original. Do not rely on translated doses or dates; check the English source.")
            malayalam_tab, english_tab = st.tabs(["മലയാളം", "English source"])
            with malayalam_tab:
                st.markdown(page.translated.replace("\n", "\n\n"))
                if page.numbers_match:
                    speak(page.translated, "ml", f"online-page-{page.page}", cached)
            with english_tab:
                st.markdown(page.original.replace("\n", "\n\n"))
    st.divider()
    st.header("Ask your report · റിപ്പോർട്ടിനോട് ചോദിക്കൂ")
    st.caption("Generative AI: Qwen answers from the English report. Google Translate handles Malayalam questions and answers. Answers include the source lines.")
    question = st.text_area("Your question in English or Malayalam", key="hybrid_question",
                            placeholder="What medicines are listed? / മരുന്നുകൾ ഏതൊക്കെയാണ്?")
    if st.button("Ask Qwen · ചോദിക്കൂ", type="primary", disabled=not question.strip()):
        if hasattr(cached, "qa_answer"):
            del cached.qa_answer
        translator = GoogleWebTranslator()
        try:
            with st.spinner("Qwen is reading your report and preparing a sourced answer…"):
                client = OllamaClient(settings.ollama_model, settings.ollama_base_url)
                client.check_ready()
                cached.qa_answer = answer_from_report(question, cached.pages, client, translator,
                                                      getattr(cached, "history", []))
                cached.qa_number = getattr(cached, "qa_number", 0) + 1
                cached.qa_question = question
                if cached.qa_answer.status in {"answered", "source_excerpt"}:
                    cached.history = (getattr(cached, "history", []) + [
                        {"role": "user", "text": question},
                        {"role": "assistant", "text": cached.qa_answer.english}])[-8:]
        except (LocalModelError, OnlineTranslationError, ValueError) as exc:
            st.error(str(exc))
        finally:
            translator.close()
    if hasattr(cached, "qa_answer"):
        result = cached.qa_answer
        with st.container(border=True):
            st.subheader("Answer · ഉത്തരം")
            st.caption(cached.qa_question)
            if result.status == "source_excerpt":
                st.info("Qwen's paraphrase could not be verified. This is a translation of the cited report excerpt; it may not fully answer your question.")
            if result.status in {"translation_warning", "unverified"}:
                st.warning(result.malayalam)
            else:
                st.write(result.malayalam)
            with st.expander("English answer", expanded=result.status == "translation_warning"):
                st.write(result.english)
            source_display(result.sources, expanded=True)
            if result.status in {"answered", "source_excerpt"}:
                speak(result.malayalam, "ml", f"hybrid-answer-{cached.qa_number}", cached)
    st.stop()

if uploaded is not None:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    current: SessionDocument | None = st.session_state.get("document")
    if current is None or current.digest != digest:
        if current:
            current.close()
            del st.session_state["document"]
        for key in ("question_text", "last_audio_hash", "last_answer", "answer_number"):
            st.session_state.pop(key, None)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            st.error("ഫയൽ വളരെ വലുതാണ്. / File is too large.")
            st.stop()
        if settings.llm_backend == "anthropic" and not settings.anthropic_api_key:
            st.error("ANTHROPIC_API_KEY .env-ൽ ചേർക്കുക. / Add your Anthropic API key to .env.")
            st.stop()
        document = None
        preview = st.empty()
        started = time.perf_counter()
        try:
            with st.status("Preparing your report…", expanded=True) as status:
                progress = st.progress(0, text="Checking the local model")
                client = make_client(settings)
                progress.progress(5, text="Reading PDF text / scanning pages")
                pages = extract_document(data, uploaded.name, client)
                chunks, lines = split_pages(pages)
                if not chunks:
                    raise IngestError("No readable report lines were found.")
                st.write(f"Read {len(pages)} page(s). Preparing searchable report text…")
                if sum(len(chunk.text) for chunk in chunks) <= MAX_DIRECT_CHARACTERS:
                    progress.progress(15, text="Short report ready — no search model download needed")
                    document = SessionDocument(digest, None, client, chunks)
                else:
                    st.caption("This longer report needs the PDF search model. Its first use downloads the model; later reports reuse it.")
                    progress.progress(15, text="Preparing the language model and report index")
                    embedder = get_embedder()
                    document = SessionDocument(digest, embedder, client)
                    document.store.add(chunks, embedder.passages([x.text for x in chunks]))
                document.lines = lines
                progress.progress(25, text="Preparing the first verified section")
                for section in iter_explanation(document.retriever, client, lines, settings.retrieval_k):
                    document.explanation.append(section)
                    progress.progress(25 + int(75 * len(document.explanation) / 6),
                                      text=f"{len(document.explanation)} of 6 sections checked")
                    with preview.container():
                        show_explanation(document, interactive=False)
                st.session_state["document"] = document
                st.session_state["processing_seconds"] = time.perf_counter() - started
                status.update(label="Your report is ready", state="complete", expanded=False)
                preview.empty()
                LOGGER.info("Document indexed: %d chunks", len(chunks))
        except ClaudeBillingError:
            st.error("Claude API ഉപയോഗിക്കാൻ ക്രെഡിറ്റ് ബാലൻസ് പോരാ. Claude Console-ൽ API credits ചേർക്കുക. / Claude API credit balance is too low. Add API credits in Claude Console.")
            st.stop()
        except LocalModelError as exc:
            st.error(f"Local model unavailable. {exc}")
            st.stop()
        except Exception as exc:
            LOGGER.warning("Document processing failed: %s", type(exc).__name__)
            st.error("റിപ്പോർട്ട് വായിക്കാനായില്ല. വ്യക്തമായ ഫയൽ നൽകി വീണ്ടും ശ്രമിക്കുക. / Could not process the report. Try a clearer file or check API access.")
            st.caption(str(exc) if isinstance(exc, (IngestError, ValueError)) else "")
            st.stop()
        finally:
            if document is not None and st.session_state.get("document") is not document:
                document.close()

document = st.session_state.get("document")
if document:
    st.caption(f"Report ready · {len(document.lines)} source lines · Prepared in {st.session_state.get('processing_seconds', 0):.0f}s")
    show_explanation(document)
    st.divider()
    st.header("ചോദിക്കൂ · Ask a question")
    language = st.radio("Question language · ചോദ്യത്തിന്റെ ഭാഷ", ["ml", "en"],
                        format_func=lambda x: "മലയാളം" if x == "ml" else "English", horizontal=True)
    audio = st.audio_input("Record a question · ചോദ്യം പറയൂ")
    if audio is not None:
        audio_hash = hashlib.sha256(audio.getvalue()).hexdigest()
        if st.session_state.get("last_audio_hash") != audio_hash:
            try:
                with st.spinner("ശബ്ദം എഴുത്താക്കുന്നു… / Transcribing…"):
                    st.session_state["question_text"] = transcribe_audio(audio.getvalue(), language, get_whisper())
                    st.session_state["last_audio_hash"] = audio_hash
            except Exception:
                st.error("ശബ്ദം മനസ്സിലാക്കാനായില്ല. വീണ്ടും പറയുക, അല്ലെങ്കിൽ എഴുതുക. / Could not transcribe. Try again or type your question.")
    st.caption("Microphone blocked? Allow access in your browser, or type below. · മൈക്ക് ലഭ്യമല്ലെങ്കിൽ താഴെ എഴുതുക.")
    question = st.text_area("Edit transcript or type a question · ചോദ്യം തിരുത്തുകയോ എഴുതുകയോ ചെയ്യൂ",
                            key="question_text", height=100)
    if st.button("Ask · ചോദിക്കൂ", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("റിപ്പോർട്ടിൽ ഉത്തരം തിരയുന്നു… / Finding answer…"):
                answer = answer_question(question, language, document.retriever, document.client,
                                         document.lines, document.history, settings.retrieval_k)
            document.history.extend([{"role": "user", "text": question},
                                     {"role": "assistant", "text": answer.text}])
            document.history = document.history[-8:]
            st.session_state["last_answer"] = answer
            st.session_state["answer_number"] = st.session_state.get("answer_number", 0) + 1
        except ClaudeBillingError:
            st.error("Claude API credit balance is too low. Add API credits in Claude Console. / Claude Console-ൽ API credits ചേർക്കുക.")
        except LocalModelError as exc:
            st.error(f"Local model unavailable. {exc}")
        except Exception:
            LOGGER.warning("Answering failed", exc_info=False)
            st.error("ഇപ്പോൾ ഉത്തരം നൽകാനായില്ല. വീണ്ടും ശ്രമിക്കുക. / Could not answer now. Please retry.")
    if "last_answer" in st.session_state:
        show_answer(st.session_state["last_answer"], document, st.session_state.get("answer_number", 0))
