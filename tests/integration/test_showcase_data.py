"""The skills-development showcase goes through the same governed pipeline as any other data."""

from typing import Any

from httpx import AsyncClient

from .conftest import SignIn


async def _get(client: AsyncClient, url: str, headers: dict[str, str]) -> dict[str, Any]:
    response = await client.get(url, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_learner_records_are_cleaned_masked_and_approved(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    dataset = await _get(client, "/api/v1/datasets/learner-progress", analyst)
    assert dataset["status"] == "approved" and dataset["layer"] == "silver" and dataset["queryable"] is True
    assert dataset["classification"] == "restricted-personal"
    assert {"full_name", "cell_number", "id_number"} <= set(dataset["piiColumns"])

    preview = await _get(client, "/api/v1/datasets/learner-progress/rows?limit=20", analyst)
    columns = preview["columns"]
    # ID numbers never reach the clean table; names and numbers stay masked; the learner number is a pseudonym.
    assert "id_number" not in columns
    rows = [dict(zip(columns, row, strict=True)) for row in preview["rows"]]
    assert rows and all("•" in str(row["full_name"]) for row in rows)
    assert all(not str(row["learner_id"]).startswith("LR-") for row in rows)
    assert all(row["outcome"] in ("Completed", "Dropped out", "In progress") for row in rows)


async def test_monthly_progress_is_approved_for_everyone(client: AsyncClient, sign_in: SignIn) -> None:
    executive = await sign_in("executive")
    dataset = await _get(client, "/api/v1/datasets/learner-progress-monthly", executive)
    assert dataset["status"] == "approved" and dataset["queryable"] is True
    assert dataset["rows"] == 34 * 3 * 9 * 6


async def test_the_assistant_answers_the_dropout_rate_from_the_governed_metric(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    response = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "What is the drop-out rate by programme type?", "datasetId": "learner-progress"},
    )
    assert response.status_code == 200, response.text
    turn = response.json()
    assert turn["refused"] is False, turn
    assert turn["build"]["datasetId"] == "learner-progress"
    assert turn["build"]["measure"] == "dropout_rate", turn["build"]
    assert "programme_type" in turn["sql"]
