"""Shared pieces for use cases: who is acting, permission checks, visibility rules and wording helpers."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Forbidden
from ..core.security import utcnow
from ..domain.roles import ROLE_LABELS, Permission, Role, has_permission
from ..infrastructure.db.models import Dataset, Share, User


@dataclass(frozen=True)
class Actor:
    """The member making a request, resolved from their token and re-read from the database."""

    id: str
    name: str
    email: str
    role: str
    team: str | None
    verified: bool
    mfa_enabled: bool
    token_type: str = "access"
    request_id: str | None = None
    ip: str | None = None

    @property
    def member_id(self) -> str | None:
        """The user id for records that reference a member; None for the system and embedded viewers."""
        return None if self.id in ("system", "embed") else self.id

    @property
    def role_label(self) -> str:
        return ROLE_LABELS[Role(self.role)]

    def can(self, permission: Permission) -> bool:
        return has_permission(self.role, permission)

    def require(self, permission: Permission, message: str | None = None) -> None:
        if not self.can(permission):
            raise Forbidden(message or f"Your role ({self.role_label}) cannot do this.")

    def require_verified(self) -> None:
        if not self.verified:
            raise Forbidden("Verify your email address before analysing data.")

    @classmethod
    def from_user(cls, user: User, **extra: str | None) -> "Actor":
        return cls(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role,
            team=user.team,
            verified=user.email_verified_at is not None,
            mfa_enabled=user.mfa_enabled,
            **extra,  # type: ignore[arg-type]
        )


SYSTEM = Actor(
    id="system",
    name="System",
    email="system@dataforge.local",
    role="admin",
    team=None,
    verified=True,
    mfa_enabled=False,
)


def initials(name: str) -> str:
    return "".join(part[0].upper() for part in name.split()[:2] if part) or "?"


def slugify(text: str, max_length: int = 60) -> str:
    normalised = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalised.lower()).strip("-")
    return slug[:max_length].strip("-") or "item"


def snake(text: str, max_length: int = 50) -> str:
    return slugify(text, max_length).replace("-", "_") or "item"


def humanize(moment: datetime | None, now: datetime | None = None) -> str:
    """ "2 hours ago", "Yesterday", "3 days ago", the way the frontend words times."""
    if moment is None:
        return "Never"
    now = now or utcnow()
    seconds = (now - moment).total_seconds()
    if seconds < 60:
        return "Just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(minutes // 60)
    if hours < 24 and moment.date() == now.date():
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = (now.date() - moment.date()).days
    if days <= 1:
        return "Yesterday" if days == 1 else "Today"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "Last week"
    if days < 60:
        return f"{days // 7} weeks ago"
    return moment.strftime("%d %b %Y")


async def shared_ids(session: AsyncSession, actor: Actor, resource_type: str) -> set[str]:
    rows = await session.execute(
        select(Share.resource_id).where(Share.resource_type == resource_type, Share.user_id == actor.id)
    )
    return {r[0] for r in rows}


def can_see(
    actor: Actor, owner_id: str | None, visibility: str, team: str | None, shared: set[str], resource_id: str
) -> bool:
    if owner_id == actor.id or visibility == "organisation" or resource_id in shared:
        return True
    return visibility == "team" and team is not None and team == actor.team


def can_see_dataset(actor: Actor, dataset: Dataset, shared: set[str]) -> bool:
    if dataset.erased_at is not None:
        return False
    if actor.can(Permission.VIEW_ALL_DATASETS):
        return True
    return can_see(actor, dataset.owner_id, dataset.visibility, dataset.team, shared, dataset.id)


def is_queryable(dataset: Dataset) -> bool:
    return dataset.status == "approved" and dataset.layer in ("silver", "gold") and dataset.erased_at is None
