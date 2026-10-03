"""The assistant against the seeded organisation: explaining changes, testing relationships, follow-ups."""

from typing import Any

from httpx import AsyncClient

from .conftest import SignIn


async def _ask(client: AsyncClient, headers: dict[str, str], question: str, **extra: Any) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/assistant/ask", headers=headers, json={"question": question, **extra}
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_the_assistant_explains_a_change_in_a_total(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    turn = await _ask(client, analyst, "Why did revenue change last month?")
    assert turn["refused"] is False, turn
    assert turn["build"]["method"] == "explain" and turn["sql"]
    assert "of the change" in turn["answer"]


async def test_the_assistant_tests_a_relationship(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    turn = await _ask(client, analyst, "Is average fee related to enrolments?")
    assert turn["refused"] is False, turn
    assert turn["build"]["method"] == "relationship"
    assert "relationship" in turn["answer"] and "not proof" in turn["explanation"]


async def test_a_follow_up_builds_on_the_members_own_answer(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    first = await _ask(client, analyst, "What was the average fee by region last quarter?")
    assert first["refused"] is False
    follow = await _ask(client, analyst, "what about Gauteng?", contextId=first["id"])
    assert follow["refused"] is False and follow["followsUp"] is True, follow
    assert follow["build"]["measure"] == "avg_fee" and "Gauteng" in follow["sql"]
    # The revenue dataset has no course column, so this follow-up has nothing to group by.
    assert (await _ask(client, analyst, "and by course?", contextId=first["id"]))["refused"] is True

    # Another member cannot build on someone else's conversation: the context is ignored.
    practitioner = await sign_in("practitioner")
    borrowed = await _ask(client, practitioner, "what about Gauteng?", contextId=first["id"])
    assert borrowed["refused"] is True
