"""Administration: members, roles and quotas, and row-level security policies."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Conflict, NotFound, ValidationFailed
from ..core.security import hash_password, new_opaque_token
from ..domain.roles import ROLE_LABELS, Permission, Role
from ..domain.sql_guard import QueryRefused, validate_predicate
from ..infrastructure import email as mail
from ..infrastructure.db.models import Dataset, RowPolicy, User, new_id
from . import audit, quotas
from .common import Actor, humanize
from .datasets import sync_superset_later
from .identity import issue_email_token, normalise_email


async def member_dict(session: AsyncSession, user: User) -> dict[str, Any]:
    q = await quotas.summary(session, user)
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": ROLE_LABELS[Role(user.role)],
        "roleId": user.role,
        "team": user.team,
        "status": user.status,
        "verified": user.email_verified_at is not None,
        "mfaEnabled": user.mfa_enabled,
        "lastActive": humanize(user.last_active_at),
        "quotas": q["short"],
        "usage": q["used"],
        "limits": q["limits"],
    }


async def list_members(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = await session.execute(select(User).where(User.erased_at.is_(None)).order_by(User.created_at))
    members = [await member_dict(session, u) for u in rows.scalars()]
    await session.commit()
    if actor.can(Permission.ADMIN_USERS):
        return members
    # Others see who is in the organisation (to share with them), but not quotas or security settings.
    return [
        {k: m[k] for k in ("id", "name", "email", "role", "roleId", "team", "lastActive")} for m in members
    ]


async def _admins(session: AsyncSession) -> int:
    return int(
        (
            await session.execute(
                select(func.count()).select_from(User).where(User.role == "admin", User.status == "active")
            )
        ).scalar()
        or 0
    )


async def update_member(
    session: AsyncSession, actor: Actor, user_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    actor.require(Permission.ADMIN_USERS, "Only an administrator can change members.")
    user = await session.get(User, user_id)
    if user is None or user.erased_at is not None:
        raise NotFound("Member not found.")
    described = []
    if "role" in changes and changes["role"] != user.role:
        try:
            role = Role(changes["role"])
        except ValueError as error:
            raise ValidationFailed("Unknown role.") from error
        if user.role == "admin" and await _admins(session) <= 1:
            raise Conflict("The organisation needs at least one administrator.")
        described.append(f"role {ROLE_LABELS[Role(user.role)]} → {ROLE_LABELS[role]}")
        user.role = role.value
    if "status" in changes and changes["status"] != user.status:
        if changes["status"] not in ("active", "disabled"):
            raise ValidationFailed("Status is active or disabled.")
        if user.id == actor.id:
            raise Conflict("You cannot disable your own account.")
        if user.role == "admin" and changes["status"] == "disabled" and await _admins(session) <= 1:
            raise Conflict("The organisation needs at least one administrator.")
        user.status = changes["status"]
        described.append(f"status {user.status}")
    if "team" in changes:
        user.team = (str(changes["team"]).strip() or None) if changes["team"] is not None else None
        described.append(f"team {user.team or 'none'}")
    if "quotas" in changes:
        requested = {
            k: v
            for k, v in (changes["quotas"] or {}).items()
            if k in ("computeSeconds", "storageBytes", "aiQueries")
        }
        if any(v is not None and (not isinstance(v, int) or v < 0) for v in requested.values()):
            raise ValidationFailed("Quotas are whole numbers, or empty for the role's default.")
        # An empty value removes the override. It is never stored: a stored null would read as "unlimited".
        overrides = {k: v for k, v in (user.quota_overrides or {}).items() if v is not None}
        for key, value in requested.items():
            if value is None:
                overrides.pop(key, None)
            else:
                overrides[key] = value
        user.quota_overrides = overrides or None
        described.append("quotas")
    await audit.record(
        session, actor, "Changed member", "user", user.id, user.email, ", ".join(described) or "no change"
    )
    await session.commit()
    return await member_dict(session, user)


async def invite(
    session: AsyncSession, actor: Actor, name: str, email: str, role: str, team: str | None
) -> dict[str, Any]:
    """Create a member and email them a link to choose their password."""
    actor.require(Permission.ADMIN_USERS, "Only an administrator can invite members.")
    email = normalise_email(email)
    if (await session.execute(select(User).where(User.email == email))).scalar_one_or_none():
        raise Conflict("A member with this email already exists.")
    try:
        chosen = Role(role)
    except ValueError as error:
        raise ValidationFailed("Unknown role.") from error
    user = User(
        id=new_id(),
        name=" ".join(name.split()) or email,
        email=email,
        role=chosen.value,
        team=team,
        password_hash=hash_password(new_opaque_token()),
    )
    session.add(user)
    await session.flush()
    token = await issue_email_token(session, user, "reset", 72)
    await audit.record(session, actor, "Invited member", "user", user.id, user.email, ROLE_LABELS[chosen])
    await session.commit()
    await mail.send_password_reset(user.email, user.name, token)
    return await member_dict(session, user)


# ------------------------------------------------------------------ row-level security


async def list_policies(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    actor.require(Permission.MANAGE_POLICIES, "Only a Data Steward manages row policies.")
    rows = await session.execute(
        select(RowPolicy, Dataset.name)
        .join(Dataset, Dataset.id == RowPolicy.dataset_id)
        .order_by(Dataset.name, RowPolicy.role)
    )
    return [
        {
            "id": p.id,
            "datasetId": p.dataset_id,
            "dataset": name,
            "role": p.role,
            "roleLabel": ROLE_LABELS[Role(p.role)],
            "predicate": p.predicate,
            "description": p.description,
        }
        for p, name in rows
    ]


async def set_policy(
    session: AsyncSession, actor: Actor, dataset_id: str, role: str, predicate: str, description: str
) -> dict[str, Any]:
    actor.require(Permission.MANAGE_POLICIES, "Only a Data Steward manages row policies.")
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None or dataset.erased_at is not None or not dataset.fields:
        raise NotFound("Row policies apply to approved datasets.")
    try:
        Role(role)
    except ValueError as error:
        raise ValidationFailed("Unknown role.") from error
    columns = {c["name"] for c in dataset.fields.get("columns", [])}
    try:
        cleaned = validate_predicate(predicate, columns)
    except QueryRefused as error:
        raise ValidationFailed(f"{error.reason} {error.suggestion}".strip()) from error
    existing = (
        await session.execute(
            select(RowPolicy).where(RowPolicy.dataset_id == dataset_id, RowPolicy.role == role)
        )
    ).scalar_one_or_none()
    if existing is None:
        existing = RowPolicy(
            dataset_id=dataset_id, role=role, predicate=cleaned, description=description, created_by=actor.id
        )
        session.add(existing)
    else:
        existing.predicate, existing.description = cleaned, description
    await session.flush()
    await audit.record(
        session,
        actor,
        "Set row policy",
        "dataset",
        dataset.id,
        dataset.name,
        f"{ROLE_LABELS[Role(role)]}: {cleaned}",
    )
    await session.commit()
    await sync_superset_later(dataset.id)
    return {
        "id": existing.id,
        "datasetId": dataset_id,
        "role": role,
        "predicate": cleaned,
        "description": description,
    }


async def delete_policy(session: AsyncSession, actor: Actor, policy_id: str) -> None:
    actor.require(Permission.MANAGE_POLICIES, "Only a Data Steward manages row policies.")
    policy = await session.get(RowPolicy, policy_id)
    if policy is None:
        raise NotFound("Policy not found.")
    await session.delete(policy)
    await audit.record(
        session, actor, "Removed row policy", "dataset", policy.dataset_id, policy.role, policy.predicate
    )
    await session.commit()
    await sync_superset_later(policy.dataset_id)
