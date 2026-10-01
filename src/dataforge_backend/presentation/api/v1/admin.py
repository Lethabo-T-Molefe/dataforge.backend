from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import admin
from ....application.common import Actor, humanize
from ....domain.roles import DEFAULT_QUOTAS, ROLE_LABELS, ROLE_PERMISSIONS, Permission, Role
from ....infrastructure.db.models import SandboxSession, User
from ..deps import current_actor, get_db
from ..schemas import InviteIn, MemberUpdateIn, PolicyIn

router = APIRouter(tags=["Administration"])


@router.get("/roles")
async def roles(actor: Actor = Depends(current_actor)) -> list[dict[str, Any]]:
    """Roles, what each can do, and their default monthly quotas."""
    return [
        {
            "id": role.value,
            "label": ROLE_LABELS[role],
            "permissions": sorted(p.value for p in ROLE_PERMISSIONS[role]),
            "quotas": {
                "computeSeconds": DEFAULT_QUOTAS[role].compute_seconds,
                "storageBytes": DEFAULT_QUOTAS[role].storage_bytes,
                "aiQueries": DEFAULT_QUOTAS[role].ai_queries,
            },
        }
        for role in Role
    ]


@router.get("/members")
async def members(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await admin.list_members(session, actor)


@router.post("/members", status_code=status.HTTP_201_CREATED)
async def invite(
    body: InviteIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await admin.invite(session, actor, body.name, str(body.email), body.role, body.team)


@router.patch("/members/{user_id}")
async def update_member(
    user_id: str,
    body: MemberUpdateIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await admin.update_member(session, actor, user_id, body.model_dump(exclude_unset=True))


@router.get("/policies")
async def policies(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await admin.list_policies(session, actor)


@router.put("/policies")
async def set_policy(
    body: PolicyIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Row-level security: the rows of a dataset one role may read, as a condition such as region = 'Gauteng'."""
    return await admin.set_policy(
        session, actor, body.dataset_id, body.role, body.predicate, body.description
    )


@router.delete("/policies/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_policy(
    policy_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await admin.delete_policy(session, actor, policy_id)


@router.get("/sandbox/sessions")
async def sandbox_sessions(
    limit: int = 50, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    """Recent sandbox runs with their caps and outcomes. Members see their own; auditors and admins see all."""
    stmt = (
        select(SandboxSession, User.name)
        .outerjoin(User, User.id == SandboxSession.user_id)
        .order_by(SandboxSession.started_at.desc())
        .limit(min(limit, 500))
    )
    if not actor.can(Permission.VIEW_ALL_AUDIT):
        stmt = stmt.where(SandboxSession.user_id == actor.id)
    rows = await session.execute(stmt)
    return [
        {
            "id": s.id,
            "member": name or "System",
            "kind": s.kind,
            "status": s.status,
            "caps": s.caps,
            "rows": s.rows_returned,
            "durationMs": s.duration_ms,
            "error": s.error,
            "when": humanize(s.started_at),
            "startedAt": s.started_at.isoformat(),
        }
        for s, name in rows
    ]
