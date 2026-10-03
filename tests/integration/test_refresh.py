"""Refreshing a connected source: the saved recipe reruns, approval is kept only when nothing changed shape,
and the people who need to know are told."""

import io
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from dataforge_backend.core.database import engine

from .conftest import SignIn

SOURCE = "refresh_source"


async def _source(*statements: str) -> None:
    async with engine.begin() as conn:
        for statement in statements:
            await conn.execute(text(statement))


@pytest.fixture
async def source() -> AsyncIterator[None]:
    """A small source table in the test database itself, read back through a real connector."""
    rows = ", ".join(
        f"('{['Gauteng', 'gauteng', 'Limpopo'][i % 3]}', '2026-0{1 + i % 3}', {100 + i})" for i in range(30)
    )
    await _source(
        f"DROP SCHEMA IF EXISTS {SOURCE} CASCADE",
        f"CREATE SCHEMA {SOURCE}",
        f"CREATE TABLE {SOURCE}.fees (region text, month text, fee integer)",
        f"INSERT INTO {SOURCE}.fees VALUES {rows}",
    )
    yield
    await _source(f"DROP SCHEMA IF EXISTS {SOURCE} CASCADE")


async def _ok(response: Any, status: int = 200) -> Any:
    assert response.status_code == status, response.text
    return response.json()


async def test_a_refresh_keeps_approval_only_when_nothing_changed_shape(
    client: AsyncClient, sign_in: SignIn, source: None
) -> None:
    steward = await sign_in("steward")
    url = engine.url
    connector = await _ok(
        await client.post(
            "/api/v1/connectors",
            headers=steward,
            json={
                "name": "Refresh test source",
                "kind": "postgres",
                "host": url.host,
                "port": url.port or 5432,
                "database": url.database,
                "schema": SOURCE,
                "user": url.username,
                "password": url.password or "",
                "sslmode": "disable",
            },
        ),
        201,
    )
    assert connector["status"] == "connected", connector
    dataset = await _ok(
        await client.post(
            f"/api/v1/connectors/{connector['id']}/import",
            headers=steward,
            json={"table": "fees", "name": "Refresh fees", "visibility": "organisation"},
        ),
        201,
    )
    dataset_id = dataset["id"]
    await _ok(
        await client.post(
            f"/api/v1/cleaning/{dataset_id}/recipe/steps", headers=steward, json={"methodId": "cat-map"}
        ),
        201,
    )
    await _ok(await client.post(f"/api/v1/cleaning/{dataset_id}/run", headers=steward))
    await _ok(
        await client.post(
            f"/api/v1/datasets/{dataset_id}/approve", headers=steward, json={"reason": "Checked"}
        )
    )

    # Nothing changed at the source: no new raw version.
    unchanged = await _ok(await client.post(f"/api/v1/connectors/{connector['id']}/sync", headers=steward))
    assert [o["status"] for o in unchanged] == ["unchanged"]

    # New rows, same shape: the recipe reruns and the dataset stays approved and queryable.
    await _source(
        f"INSERT INTO {SOURCE}.fees VALUES ('Limpopo', '2026-03', 400), ('gauteng', '2026-03', 410)"
    )
    current = await _ok(await client.post(f"/api/v1/connectors/{connector['id']}/sync", headers=steward))
    assert [o["status"] for o in current] == ["current"], current
    after = await _ok(await client.get(f"/api/v1/datasets/{dataset_id}", headers=steward))
    assert after["queryable"] is True and after["rows"] == 32 and after["version"] == 2
    count = await _ok(
        await client.post(
            "/api/v1/sql/run", headers=steward, json={"sql": f"SELECT COUNT(*) AS n FROM {after['table']}"}
        )
    )
    assert count["rows"] == [[32]]

    # A new column changes the shape: back to a steward, who is told.
    await _source(f"ALTER TABLE {SOURCE}.fees ADD COLUMN note text")
    review = await _ok(await client.post(f"/api/v1/connectors/{connector['id']}/sync", headers=steward))
    assert [o["status"] for o in review] == ["review"], review
    pending = await _ok(await client.get(f"/api/v1/datasets/{dataset_id}", headers=steward))
    assert pending["queryable"] is False and pending["status"] == "pending"

    inbox = await _ok(await client.get("/api/v1/notifications", headers=steward))
    titles = [n["title"] for n in inbox["items"]]
    assert "Refresh fees needs review after a refresh" in titles
    assert inbox["unread"] >= 1
    read = await _ok(await client.post("/api/v1/notifications/read", headers=steward, json={}))
    assert read["unread"] == 0

    # Only a steward schedules refreshes, and only the requester or a steward refreshes now.
    analyst = await sign_in("analyst")
    refused = await client.patch(
        f"/api/v1/connectors/{connector['id']}", headers=analyst, json={"refreshCadence": "daily"}
    )
    assert refused.status_code == 403
    assert (
        await client.post(f"/api/v1/connectors/{connector['id']}/sync", headers=analyst)
    ).status_code == 403
    scheduled = await _ok(
        await client.patch(
            f"/api/v1/connectors/{connector['id']}", headers=steward, json={"refreshCadence": "daily"}
        )
    )
    assert scheduled["refreshCadence"] == "daily"


async def test_owners_hear_about_decisions_on_their_data(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    before = await _ok(await client.get("/api/v1/notifications", headers=analyst))
    owned = await _ok(
        await client.post(
            "/api/v1/datasets",
            headers=analyst,
            files={
                "file": ("fees_draft.csv", io.BytesIO(b"region,fee\nGauteng,100\nLimpopo,90\n"), "text/csv")
            },
            data={"name": "Fees draft", "visibility": "organisation"},
        ),
        201,
    )
    steward = await sign_in("steward")
    await _ok(
        await client.post(
            f"/api/v1/datasets/{owned['id']}/reject", headers=steward, json={"reason": "Fix the dates"}
        )
    )
    after = await _ok(await client.get("/api/v1/notifications", headers=analyst))
    newest = after["items"][0]
    assert newest["kind"] == "decision" and "Fix the dates" in newest["body"]
    assert after["unread"] == before["unread"] + 1
