"""Local Ollama or optional Claude generation, plus groundedness checks."""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import anthropic
import httpx
from PIL import Image


PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class ClaudeBillingError(RuntimeError):
    """Claude Console has insufficient API credits for the request."""


class LocalModelError(RuntimeError):
    """The local Ollama service or requested model is unavailable."""


class TextClient(Protocol):
    def text(self, system: str, user: str, max_tokens: int = 1200) -> str: ...
    def json(self, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]: ...


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 transport: httpx.BaseTransport | None = None) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("OLLAMA_BASE_URL must point to local HTTP Ollama")
        self.model = model
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=300,
                                   transport=transport, trust_env=False)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise LocalModelError("Ollama is not running. Install Ollama, then open it.") from exc
        except httpx.TimeoutException as exc:
            raise LocalModelError("Ollama timed out. Try again after the model finishes loading.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise LocalModelError(f"Model {self.model} is missing. Run: ollama pull {self.model}") from exc
            raise LocalModelError(f"Ollama returned HTTP {exc.response.status_code}.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LocalModelError("Could not read the Ollama response.") from exc

    def check_ready(self) -> None:
        models = self._request("GET", "/api/tags").get("models", [])
        if not any(item.get("name") == self.model or item.get("model") == self.model for item in models):
            raise LocalModelError(f"Model {self.model} is missing. Run: ollama pull {self.model}")

    def text(self, system: str, user: str, max_tokens: int = 1200) -> str:
        payload = self._request("POST", "/api/chat", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False, "think": False, "keep_alive": "15m",
            "options": {"temperature": 0, "num_predict": max_tokens},
        })
        content = payload.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise LocalModelError("Ollama returned an empty answer.")
        return content

    def json(self, system: str, user: str, max_tokens: int = 1200,
             schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self._request("POST", "/api/chat", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False, "think": False, "format": schema or "json", "keep_alive": "15m",
            "options": {"temperature": 0, "num_predict": max_tokens},
        })
        raw = payload.get("message", {}).get("content", "")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Ollama returned a non-object JSON value")
        return value


def make_client(settings: Any) -> TextClient:
    if settings.llm_backend == "ollama":
        client = OllamaClient(settings.ollama_model, settings.ollama_base_url)
        client.check_ready()
        return client
    if settings.llm_backend == "anthropic":
        return ClaudeClient(settings.anthropic_api_key, settings.claude_model)
    raise ValueError("LLM_BACKEND must be ollama or anthropic")


class ClaudeClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def text(self, system: str, user: str, max_tokens: int = 1200) -> str:
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=0,
                system=system, messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as exc:
            error = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
            message = error.get("message", "") if isinstance(error, dict) else ""
            if "credit balance is too low" in message.lower():
                raise ClaudeBillingError("Claude API credit balance is too low") from exc
            raise
        return "".join(block.text for block in response.content if block.type == "text")

    def json(self, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]:
        raw = self.text(system, user, max_tokens)
        # Claude occasionally wraps valid JSON in a Markdown fence.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Claude returned a non-object JSON value")
        return value

    def transcribe_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = self.client.messages.create(
            model=self.model, max_tokens=4000, temperature=0,
            system="Transcribe every visible English word and number exactly. Preserve line breaks and headings. Do not explain or infer missing text. If unreadable, mark [unreadable].",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}},
                {"type": "text", "text": "Transcribe this hospital document."},
            ]}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def render_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"CHUNK {x['id']} | page {x['page']} | section {x['section']}\n{x['text']}"
        for x in chunks
    )


_VERIFY_LOGGER = logging.getLogger(__name__ + ".verify")


def verify_claim(client: TextClient, claim: str, evidence: str) -> bool:
    """Check semantic support; an invalid audit never establishes support."""
    is_ollama = isinstance(client, OllamaClient)
    try:
        kwargs: dict[str, Any] = {}
        if is_ollama:
            kwargs["schema"] = {"type": "object", "additionalProperties": False,
                                "properties": {"supported": {"type": "boolean"}},
                                "required": ["supported"]}
        result = client.json(
            "You are a strict source-grounding auditor. Return only JSON: {\"supported\": boolean}. "
            "Mark false if ANY clinical detail, instruction, number, or interpretation in the claim is absent from the English evidence. "
            "Treat Malayalam and English as equivalent languages. Do not add medical knowledge.",
            f"English evidence:\n{evidence}\n\nClaim:\n{claim}", max_tokens=80, **kwargs,
        )
        supported = result.get("supported")
        if supported is not True:
            _VERIFY_LOGGER.warning("Grounding audit did not confirm support")
        return supported is True
    except (anthropic.APIError, LocalModelError):
        raise
    except Exception as exc:
        _VERIFY_LOGGER.warning("Grounding audit failed: %s", type(exc).__name__)
        return False
