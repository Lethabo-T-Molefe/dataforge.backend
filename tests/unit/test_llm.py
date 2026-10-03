"""The model gateway's request and its handling of every way a response can go wrong, with a fake client."""

import json
from types import SimpleNamespace
from typing import Any

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
    with pytest.raises(llm.PlanUnavailable):
        await llm.propose_plan("q", "c", {})
