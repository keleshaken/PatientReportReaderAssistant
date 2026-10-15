"""Environment-backed application settings."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    translation_backend: str = os.getenv("TRANSLATION_BACKEND", "local").lower()
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama").lower()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    session_ttl_hours: int = int(os.getenv("SESSION_TTL_HOURS", "24"))


settings = Settings()
