"""Slack: ask the assistant with a slash command, and link a Slack account to a DataForge member.

A member creates a one-time link code in DataForge, then types `/dataforge link CODE` in Slack. After that,
`/dataforge <question>` is answered with that member's own permissions.
"""

import secrets
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Forbidden
from ..core.security import hash_token, utcnow
from ..infrastructure.db.models import SlackLink, User
from . import assistant, audit
from .common import Actor

LINK_MINUTES = 15


async def create_link_code(session: AsyncSession, actor: Actor) -> dict[str, Any]:
    code = "-".join(secrets.token_hex(2).upper() for _ in range(2))
    session.add(
        SlackLink(
            user_id=actor.id,
            link_code_hash=hash_token(code),
            link_expires_at=utcnow() + timedelta(minutes=LINK_MINUTES),
        )
    )
    await audit.record(session, actor, "Started Slack link", "integration", None, "Slack")
    await session.commit()
    return {
        "code": code,
        "expiresInMinutes": LINK_MINUTES,
        "instruction": f"In Slack, type: /dataforge link {code}",
    }


def _reply(text: str, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"response_type": "ephemeral", "text": text}
    if blocks:
        body["blocks"] = blocks
    return body


async def handle_command(session: AsyncSession, team_id: str, user_id: str, text: str) -> dict[str, Any]:
    text = text.strip()
    if text.lower().startswith("link "):
        code = text[5:].strip().upper()
        link = (
            await session.execute(select(SlackLink).where(SlackLink.link_code_hash == hash_token(code)))
        ).scalar_one_or_none()
        if link is None or link.link_expires_at is None or link.link_expires_at < utcnow():
            return _reply(
                "That link code is not valid or has expired. Create a new one in DataForge settings."
            )
        previous = (
            await session.execute(
                select(SlackLink).where(
                    SlackLink.slack_team_id == team_id, SlackLink.slack_user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if previous is not None and previous.id != link.id:
            await session.delete(previous)
            await session.flush()
        link.slack_team_id, link.slack_user_id = team_id, user_id
        link.link_code_hash, link.link_expires_at = None, None
        member = await session.get(User, link.user_id)
        if member is not None:
            await audit.record(
                session,
                Actor.from_user(member),
                "Linked Slack",
                "integration",
                None,
                "Slack",
                f"Slack user {user_id}",
            )
        await session.commit()
        return _reply("Linked. Ask a question with /dataforge followed by your question.")

    link = (
        await session.execute(
            select(SlackLink).where(SlackLink.slack_team_id == team_id, SlackLink.slack_user_id == user_id)
        )
    ).scalar_one_or_none()
    member = await session.get(User, link.user_id) if link else None
    if member is None or member.status != "active":
        return _reply(
            "Link your DataForge account first: create a link code in DataForge settings, then type /dataforge link CODE."
        )
    if not text:
        return _reply("Ask a question, for example: /dataforge what is the completion rate by course?")
    try:
        turn = await assistant.ask(session, Actor.from_user(member), text, channel="slack")
    except Forbidden as error:
        return _reply(error.message)
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{turn['answer']}*"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": turn["explanation"][:2900]}]},
    ]
    if turn.get("table") and turn["table"].get("rows"):
        columns = turn["table"]["columns"]
        lines = [" | ".join(str(c) for c in columns)] + [
            " | ".join(str(v) for v in row) for row in turn["table"]["rows"][:10]
        ]
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "```" + "\n".join(lines)[:2800] + "```"}}
        )
    if turn.get("sql"):
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "Open DataForge to see the SQL that ran."}],
            }
        )
    return _reply(turn["answer"], blocks)
