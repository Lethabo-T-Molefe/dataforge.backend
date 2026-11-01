"""The model gateway's request and its handling of every way a response can go wrong, with a fake client."""

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic.types.beta import BetaTextBlock

from dataforge_backend.core.config import settings
from dataforge_backend.infrastructure import llm

PLAN = {"action": "answer", "datasetId": "cc", "measure": "revenue", "method": "compare"}


class _Messages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.calls.append(request)
        return self.response


def _response(text: str | None, stop_reason: str = "end_turn") -> Any:
    content = [] if text is None else [BetaTextBlock.model_construct(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, model="claude-opus-5-5", content=content)


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> _Messages:
    fake = _Messages(_response(json.dumps(PLAN)))
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(settings, "ASSISTANT_USE_LLM", True)
    monkeypatch.setattr(llm, "_get_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    return fake


async def test_the_request_carries_the_schema_catalogue_and_previous_turn(messages: _Messages) -> None:
    schema = {"type": "object"}
    plan, model = await llm.propose_plan("and by course?", "Dataset cc", schema, previous="Question: revenue")
    assert plan == PLAN and model == "claude-opus-5-5"
    request = messages.calls[0]
    assert request["model"] == settings.ASSISTANT_MODEL == "claude-opus-5-5"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert request["fallbacks"] == "default" and request["betas"] == ["server-side-fallback-2026-07-01"]
    catalogue, asked = request["messages"][0]["content"]
    assert "Dataset cc" in catalogue["text"] and catalogue["cache_control"] == {"type": "ephemeral"}
    assert asked["text"].index("<previous>") < asked["text"].index("<question>")
    assert "and by course?" in asked["text"]


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (_response(None, "refusal"), "refused"),
        (_response('{"action": "answer"', "max_tokens"), "unavailable"),
        (_response("not json"), "unavailable"),
        (_response("[1, 2]"), "unavailable"),
        (_response(None), "unavailable"),
    ],
)
async def test_unusable_responses_fall_back_or_refuse(
    messages: _Messages, response: Any, outcome: str
) -> None:
    messages.response = response
    if outcome == "refused":
        plan, _ = await llm.propose_plan("q", "c", {})
        assert plan["action"] == "refuse"
    else:
        with pytest.raises(llm.PlanUnavailable):
            await llm.propose_plan("q", "c", {})


async def test_no_key_means_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    with pytest.raises(llm.PlanUnavailable):
        await llm.propose_plan("q", "c", {})


class _Post:
    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, url: str, **request: Any) -> httpx.Response:
        self.calls.append({"url": url, **request})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _chat(content: str | None, finish_reason: str = "stop", status: int = 200) -> httpx.Response:
    choice = {"finish_reason": finish_reason, "message": {"content": content}}
    return httpx.Response(status, json={"model": "nvidia/nemotron-3.5-super-vl:free", "choices": [choice]})


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> _Post:
    fake = _Post(_chat(f"<think>by region</think>\n```json\n{json.dumps(PLAN)}\n```"))
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(settings, "ASSISTANT_USE_LLM", True)
    monkeypatch.setattr(llm.outbound, "post", fake)
    return fake


async def test_without_a_claude_key_openrouter_plans(chat: _Post) -> None:
    schema = {"type": "object"}
    plan, model = await llm.propose_plan("and by course?", "Dataset cc", schema, previous="Question: revenue")
    assert plan == PLAN and model == "nvidia/nemotron-3.5-super-vl:free"
    request = chat.calls[0]
    assert request["url"] == llm.OPENROUTER_URL and request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == settings.OPENROUTER_MODEL
    assert request["json"]["response_format"]["json_schema"]["schema"] == schema
    system, user = request["json"]["messages"]
    assert json.dumps(schema) in system["content"] and "Dataset cc" in user["content"]
    assert user["content"].index("<previous>") < user["content"].index("<question>")


@pytest.mark.parametrize(
    "response",
    [
        _chat(None, status=429),
        _chat(None, status=502),
        httpx.ConnectError("down"),
        httpx.Response(200, json={"error": {"message": "provider error"}}),
        _chat('{"action": "answer"', "length"),
        _chat("not json"),
        _chat("{not json}"),
    ],
)
async def test_unusable_openrouter_replies_fall_back(
    chat: _Post, response: httpx.Response | Exception
) -> None:
    chat.response = response
    with pytest.raises(llm.PlanUnavailable):
        await llm.propose_plan("q", "c", {})
