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


def next_step(layer: Layer, status: DatasetStatus, role: Role, *, owner_ready: bool = False) -> NextStep:
    """The single next action for a dataset, worded the same way the frontend words it.

    `owner_ready` means the member owns it and can already analyse it: cleaning and approval are optional for
    their own analysis, and only needed to share it with others.
    """
    if status == DatasetStatus.REJECTED:
        return NextStep(
            "rejected", "Rejected", "A steward rejected this dataset. The reason is on its record."
        )
    if owner_ready and status != DatasetStatus.APPROVED:
        return NextStep(
            "analyse",
            "Analyse this data",
            "You can analyse it now, with personal columns masked. Clean it and have it approved to share it.",
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


def journey(
    layer: Layer, status: DatasetStatus, source: DatasetSource, *, owner_ready: bool = False
) -> list[dict[str, str]]:
    cleaned = layer != Layer.BRONZE
    approved = status == DatasetStatus.APPROVED
    if owner_ready and not approved:
        return [
            {
                "label": "Brought in",
                "detail": "Uploaded file" if source == DatasetSource.UPLOAD else "Connected source",
                "state": "done",
            },
            {
                "label": "Analysed by you",
                "detail": "Personal columns masked",
                "state": "current",
            },
            {
                "label": "Cleaned",
                "detail": f"{layer.value} layer" if cleaned else "Optional",
                "state": "done" if cleaned else "todo",
            },
            {
                "label": "Approved and shared",
                "detail": "Optional, to share it",
                "state": "todo",
            },
        ]
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
    """Approved data in the cleaned or analytics layers can be analysed by everyone permitted to see it.

    The owner's own access before approval is decided in application.common.is_owner_queryable.
    """
    return status == DatasetStatus.APPROVED and layer in (Layer.SILVER, Layer.GOLD)
