"""Roles, the permissions each role carries, and fair-use quota defaults.

These mirror the frontend's permission checks so both sides agree on who can do what.
"""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    ANALYST = "analyst"
    PRACTITIONER = "practitioner"
    STEWARD = "steward"
    EXECUTIVE = "executive"
    ADMIN = "admin"
    AUDITOR = "auditor"
    CURATOR = "curator"


ROLE_LABELS: dict[Role, str] = {
    Role.ANALYST: "Everyday Analyst",
    Role.PRACTITIONER: "Data Practitioner",
    Role.STEWARD: "Data Steward",
    Role.EXECUTIVE: "Executive and Viewer",
    Role.ADMIN: "Administrator",
    Role.AUDITOR: "Auditor",
    Role.CURATOR: "Model Curator",
}


class Permission(StrEnum):
    UPLOAD = "upload"
    CLEAN = "clean"
    ANALYSE = "analyse"
    APPROVE = "approve"
    WRITE_CODE = "write_code"
    RUN_SQL = "run_sql"
    DEFINE_METRICS = "define_metrics"
    PUBLISH_METRICS = "publish_metrics"
    MANAGE_CONNECTORS = "manage_connectors"
    REQUEST_CONNECTORS = "request_connectors"
    MANAGE_DASHBOARDS = "manage_dashboards"
    MANAGE_MONITORS = "manage_monitors"
    PREDICT = "predict"
    CURATE_MODELS = "curate_models"
    VIEW_ALL_AUDIT = "view_all_audit"
    VIEW_ALL_DATASETS = "view_all_datasets"
    MANAGE_POLICIES = "manage_policies"
    ADMIN_USERS = "admin_users"


_EVERYDAY = {
    Permission.UPLOAD,
    Permission.CLEAN,
    Permission.ANALYSE,
    Permission.RUN_SQL,
    Permission.REQUEST_CONNECTORS,
    Permission.MANAGE_DASHBOARDS,
    Permission.MANAGE_MONITORS,
    Permission.PREDICT,
}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ANALYST: frozenset(_EVERYDAY),
    Role.PRACTITIONER: frozenset(_EVERYDAY | {Permission.WRITE_CODE, Permission.DEFINE_METRICS}),
    Role.STEWARD: frozenset(
        _EVERYDAY
        | {
            Permission.WRITE_CODE,
            Permission.DEFINE_METRICS,
            Permission.PUBLISH_METRICS,
            Permission.APPROVE,
            Permission.MANAGE_CONNECTORS,
            Permission.VIEW_ALL_AUDIT,
            Permission.VIEW_ALL_DATASETS,
            Permission.MANAGE_POLICIES,
        }
    ),
    # Executives read governed results: dashboards, answers and monitors. They do not change data.
    Role.EXECUTIVE: frozenset({Permission.ANALYSE, Permission.MANAGE_MONITORS, Permission.PREDICT}),
    Role.ADMIN: frozenset(Permission),
    # Auditors inspect the audit log and lineage and cannot alter anything.
    Role.AUDITOR: frozenset({Permission.VIEW_ALL_AUDIT, Permission.VIEW_ALL_DATASETS}),
    Role.CURATOR: frozenset(
        {
            Permission.ANALYSE,
            Permission.RUN_SQL,
            Permission.WRITE_CODE,
            Permission.PREDICT,
            Permission.CURATE_MODELS,
        }
    ),
}

READ_ONLY_ROLES = frozenset({Role.EXECUTIVE, Role.AUDITOR})


def has_permission(role: Role | str, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[Role(role)]


def is_read_only(role: Role | str) -> bool:
    return Role(role) in READ_ONLY_ROLES


@dataclass(frozen=True)
class QuotaLimits:
    """Monthly limits. None means unlimited."""

    compute_seconds: int | None
    storage_bytes: int | None
    ai_queries: int | None


_HOUR = 3600
_GB = 1024**3

DEFAULT_QUOTAS: dict[Role, QuotaLimits] = {
    Role.ANALYST: QuotaLimits(40 * _HOUR, 10 * _GB, 200),
    Role.PRACTITIONER: QuotaLimits(80 * _HOUR, 25 * _GB, 400),
    Role.STEWARD: QuotaLimits(80 * _HOUR, 50 * _GB, 400),
    Role.EXECUTIVE: QuotaLimits(4 * _HOUR, 2 * _GB, 40),
    Role.ADMIN: QuotaLimits(None, None, None),
    Role.AUDITOR: QuotaLimits(0, 0, 0),
    Role.CURATOR: QuotaLimits(40 * _HOUR, 15 * _GB, 120),
}
