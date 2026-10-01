"""Governance vocabulary and the dataset journey (brought in → cleaned → approved → analysed)."""

from dataclasses import dataclass
from enum import StrEnum

from .roles import Permission, Role, has_permission


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DatasetStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Classification(StrEnum):
    RESTRICTED_PERSONAL = "restricted-personal"
    INTERNAL_CONFIDENTIAL = "internal-confidential"
    INTERNAL_GENERAL = "internal-general"


class Visibility(StrEnum):
    PRIVATE = "private"
    TEAM = "team"
    ORGANISATION = "organisation"


class DatasetSource(StrEnum):
    UPLOAD = "upload"
    CONNECTOR = "connector"


CLASSIFICATION_LABELS: dict[Classification, str] = {
    Classification.RESTRICTED_PERSONAL: "Restricted · personal",
    Classification.INTERNAL_CONFIDENTIAL: "Internal · confidential",
    Classification.INTERNAL_GENERAL: "Internal · general",
}

LAYER_LABELS: dict[Layer, str] = {
    Layer.BRONZE: "Bronze · raw",
    Layer.SILVER: "Silver · cleaned",
    Layer.GOLD: "Gold · ready",
}


@dataclass(frozen=True)
class NextStep:
    kind: str  # clean | view | approve | wait | rejected | analyse
    label: str
    reason: str


def next_step(layer: Layer, status: DatasetStatus, role: Role) -> NextStep:
    """The single next action for a dataset, worded the same way the frontend words it."""
    if status == DatasetStatus.REJECTED:
        return NextStep(
            "rejected", "Rejected", "A steward rejected this dataset. The reason is on its record."
        )
    if layer == Layer.BRONZE:
        if has_permission(role, Permission.CLEAN):
            return NextStep(
                "clean",
                "Clean this data",
                "It is still raw. Clean it before it can be approved and analysed.",
            )
        return NextStep("view", "View the quality profile", "It is still raw and waiting to be cleaned.")
    if status == DatasetStatus.PENDING:
        if has_permission(role, Permission.APPROVE):
            return NextStep("approve", "Review and approve", "It is cleaned and waiting for your decision.")
        return NextStep(
            "wait",
            "Waiting for approval",
            "It is cleaned. A steward must approve it before it can be analysed.",
        )
    return NextStep("analyse", "Analyse this data", "It is approved and ready to analyse.")


def journey(layer: Layer, status: DatasetStatus, source: DatasetSource) -> list[dict[str, str]]:
    cleaned = layer != Layer.BRONZE
    approved = status == DatasetStatus.APPROVED
    return [
        {
            "label": "Brought in",
            "detail": "Uploaded file" if source == DatasetSource.UPLOAD else "Connected source",
            "state": "done",
        },
        {
            "label": "Cleaned",
            "detail": f"{layer.value} layer" if cleaned else "Still raw",
            "state": "done" if cleaned else "current",
        },
        {
            "label": "Approved",
            "detail": "By a steward"
            if approved
            else ("Rejected" if status == DatasetStatus.REJECTED else "Not yet"),
            "state": "done" if approved else ("current" if cleaned else "todo"),
        },
        {
            "label": "Analysed and shared",
            "detail": "Ready now" if approved else "After approval",
            "state": "current" if approved else "todo",
        },
    ]


def is_queryable(layer: Layer, status: DatasetStatus) -> bool:
    """Only approved data in the cleaned or analytics layers can be analysed."""
    return status == DatasetStatus.APPROVED and layer in (Layer.SILVER, Layer.GOLD)
