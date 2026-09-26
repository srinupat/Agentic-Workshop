"""Triage decision schema.

Epic 2's agent imports TriageDecision as its structured-output type. Any invalid
decision raises pydantic.ValidationError, which names the offending field.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class Category(StrEnum):
    BILLING = "billing"
    BUG = "bug"
    ACCESS = "access"
    PERFORMANCE = "performance"
    HOW_TO = "how-to"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Route(StrEnum):
    BILLING_TEAM = "billing-team"
    BUG_TEAM = "bug-team"
    ACCESS_TEAM = "access-team"
    PERFORMANCE_TEAM = "performance-team"
    HOW_TO_TEAM = "how-to-team"


NonEmptyRationale = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    route: Route
    rationale: NonEmptyRationale
