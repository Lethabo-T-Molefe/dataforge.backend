"""Request bodies. JSON uses camelCase, as the frontend does; Python uses snake_case."""

import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# The address syntax browsers accept for <input type="email">, with at least one dot in the domain.
_EMAIL = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)


def _email(value: str) -> str:
    """Check the shape of an address and lower-case it.

    Only the shape is checked here. Organisations run mail on internal names such as name@org.local, which
    stricter validators refuse as "not deliverable"; whether an address is real is settled by the
    verification email, not by its domain.
    """
    address = value.strip()
    if len(address) > 254 or not _EMAIL.fullmatch(address):
        raise ValueError("Enter an email address like name@organisation.org.")
    return address.lower()


Email = Annotated[str, AfterValidator(_email)]


class Body(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid", str_strip_whitespace=True
    )


class FreeBody(BaseModel):
    """Bodies whose shape is validated by the domain (analysis settings, metric formulas)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")


# ------------------------------------------------------------------ identity


class RegisterIn(Body):
    name: str = Field(min_length=1, max_length=200)
    email: Email
    password: str = Field(min_length=1, max_length=256)


class LoginIn(Body):
    email: Email
    password: str = Field(min_length=1, max_length=256)


class MfaIn(Body):
    challenge_token: str
    code: str = Field(min_length=6, max_length=10)


class CodeIn(Body):
    code: str = Field(min_length=6, max_length=10)


class TokenIn(Body):
    token: str = Field(min_length=10, max_length=400)


class RefreshIn(Body):
    refresh_token: str = Field(min_length=10, max_length=400)


class LogoutIn(Body):
    refresh_token: str | None = None


class EmailIn(Body):
    email: Email


class ResetIn(Body):
    token: str = Field(min_length=10, max_length=400)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordIn(Body):
    current_password: str
    new_password: str


class DisableMfaIn(Body):
    password: str
    code: str


class DemoIn(Body):
    role: str = "analyst"


# ------------------------------------------------------------------ administration


class MemberUpdateIn(Body):
    role: str | None = None
    status: Literal["active", "disabled"] | None = None
    team: str | None = None
    quotas: dict[str, int | None] | None = None


class InviteIn(Body):
    name: str = Field(min_length=1, max_length=200)
    email: Email
    role: str = "analyst"
    team: str | None = None


class PolicyIn(Body):
    dataset_id: str
    role: str
    predicate: str = Field(min_length=1, max_length=2000)
    description: str = ""


# ------------------------------------------------------------------ datasets


class DatasetUpdateIn(Body):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    classification: Literal["restricted-personal", "internal-confidential", "internal-general"] | None = None
    visibility: Literal["private", "team", "organisation"] | None = None
    default_measure: str | None = None


class DecisionIn(Body):
    reason: str = Field(default="", max_length=2000)


class ShareIn(Body):
    emails: list[Email] = Field(min_length=1, max_length=50)


class InsightDecisionIn(Body):
    decision: Literal["accepted", "dismissed"]


class PromoteIn(Body):
    columns: list[str] | None = None


class EraseIn(Body):
    reason: str = Field(default="Right to erasure", max_length=2000)
    confirm: str


# ------------------------------------------------------------------ cleaning


class StepIn(Body):
    method_id: str
    params: dict[str, Any] | None = None
    status: Literal["accepted", "skipped"] = "accepted"


class StepUpdateIn(Body):
    params: dict[str, Any] | None = None
    status: Literal["accepted", "skipped"] | None = None


class PreviewIn(Body):
    method_id: str
    params: dict[str, Any] | None = None


class ReorderIn(Body):
    step_ids: list[str]


class RecipeRefIn(Body):
    recipe_id: str


class VisibilityIn(Body):
    visibility: Literal["private", "team", "organisation"]


# ------------------------------------------------------------------ analysis, SQL and notebooks


class ExplorationIn(Body):
    name: str = Field(max_length=200)
    config: dict[str, Any]


class SqlIn(Body):
    sql: str = Field(min_length=1, max_length=40_000)
    max_rows: int | None = Field(default=None, ge=1, le=50_000)


class NotebookIn(Body):
    name: str = Field(max_length=200)
    cells: list[dict[str, Any]] | None = None


class NotebookUpdateIn(Body):
    name: str | None = Field(default=None, max_length=200)
    visibility: Literal["private", "team", "organisation"] | None = None


class CellIn(Body):
    kind: Literal["markdown", "sql"] = "sql"
    source: str = Field(default="", max_length=40_000)
    revision: int | None = None
    position: int | None = None


class AskIn(Body):
    question: str = Field(min_length=1, max_length=2000)


# ------------------------------------------------------------------ dashboards, monitors, predictions, connectors


class DashboardIn(Body):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    visibility: Literal["private", "team", "organisation"] = "private"


class DashboardUpdateIn(Body):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    visibility: str | None = None


class ChartIn(Body):
    title: str = Field(max_length=200)
    type: Literal["bar", "line", "area"] = "bar"
    config: dict[str, Any] | None = None
    sql: str | None = Field(default=None, max_length=40_000)
    metric: str | None = None
    explanation: str | None = Field(default=None, max_length=2000)


class EmbedIn(Body):
    enabled: bool


class ScoreIn(Body):
    model_id: str | None = None
    cohort: dict[str, str] = Field(default_factory=dict)


class ConnectorIn(Body):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["postgres", "warehouse", "spreadsheet"]
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    user: str | None = None
    password: str | None = Field(default=None, max_length=500)
    sslmode: Literal["disable", "prefer", "require", "verify-ca", "verify-full"] | None = None
    filename: str | None = None


class ImportIn(Body):
    table: str = Field(min_length=1, max_length=200)
    name: str | None = None
    description: str | None = None
    classification: Literal["restricted-personal", "internal-confidential", "internal-general"] = (
        "internal-confidential"
    )
    visibility: Literal["private", "team", "organisation"] = "private"


class ContactIn(Body):
    name: str = Field(min_length=2, max_length=200)
    email: Email
    organisation: str = Field(default="", max_length=200)
    message: str = Field(min_length=8, max_length=4000)


class OpenSqlIn(Body):
    title: str = Field(default="From DataForge", max_length=200)
    sql: str = Field(min_length=1, max_length=40_000)
    note: str = Field(default="", max_length=4000)
