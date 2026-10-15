"""Exercise the local provider without downloading a model or contacting a server."""
import json

import httpx
import pytest

from src.llm import LocalModelError, OllamaClient


def test_local_client_checks_model_and_generates_json():
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        body = json.loads(request.content)
        assert body["format"] == "json"
        assert body["think"] is False
        assert body["stream"] is False
        assert body["messages"][1]["content"] == "question"
        return httpx.Response(200, json={"message": {"content": '{"grounded": false}'}})

    client = OllamaClient("qwen3:4b-instruct", transport=httpx.MockTransport(respond))
    client.check_ready()
    assert client.json("instructions", "question") == {"grounded": False}
    assert [r.url.path for r in requests] == ["/api/tags", "/api/chat"]


def test_local_client_reports_missing_model():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []}))
    with pytest.raises(LocalModelError, match="ollama pull qwen3:4b-instruct"):
        OllamaClient("qwen3:4b-instruct", transport=transport).check_ready()


def test_local_client_refuses_remote_host():
    with pytest.raises(ValueError, match="local HTTP"):
        OllamaClient("qwen3:4b-instruct", "https://example.org")


def test_local_client_sends_structured_output_schema():
    schema = {"type": "object", "properties": {"supported": {"type": "boolean"}},
              "required": ["supported"], "additionalProperties": False}

    def respond(request):
        assert json.loads(request.content)["format"] == schema
        return httpx.Response(200, json={"message": {"content": '{"supported": true}'}})

    client = OllamaClient("qwen3:1.7b", transport=httpx.MockTransport(respond))
    assert client.json("audit", "claim", schema=schema) == {"supported": True}
