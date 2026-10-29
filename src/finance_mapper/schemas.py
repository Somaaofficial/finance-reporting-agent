from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class MappingDecision(BaseModel):
    """Validated result of one account-mapping decision."""

    model_config = ConfigDict(extra="forbid")

    source_account_code: str
    source_account_name: str | None = None

    target_account_code: str | None = None
    target_account_name: str | None = None

    status: Literal[
        "exact_match",
        "needs_review",
        "unmapped",
    ]

    method: Literal[
        "exact_code",
        "retrieval",
        "llm_recommendation",
        "human_review",
        "none",
    ]

    confidence_score: float = Field(ge=0.0, le=1.0)

    reason: str

    evidence: list[str] = Field(default_factory=list)

    requires_human_review: bool

class MappingCandidate(BaseModel):
    """A retrieved candidate, not an approved accounting decision."""

    source_account_code: str
    source_account_name: str | None = None
    source_locations: list[str] = Field(default_factory=list)
    source_context: list[str] = Field(default_factory=list)

    candidate_account_code: str
    candidate_account_name: str
    candidate_account_type: str

    similarity_score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)

    evidence: list[str] = Field(default_factory=list)

    status: Literal["candidate_only"] = "candidate_only"
    requires_human_review: bool = True


class MappingRecommendation(BaseModel):
    """LLM recommendation based only on retrieved COA candidates."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["recommend", "abstain"]

    recommended_account_code: str | None = None

    reason: str = Field(min_length=1)

    supporting_evidence: list[str] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_recommendation(self) -> "MappingRecommendation":
        code = self.recommended_account_code

        if self.decision == "recommend":
            if code is None or not code.strip():
                raise ValueError(
                    "A recommendation must include an account code."
                )

            self.recommended_account_code = code.strip()

        elif code is not None:
            raise ValueError(
                "An abstention must not include an account code."
            )

        return self

class HumanReviewDecision(BaseModel):
    """One reviewer's decision for a proposed account mapping."""

    model_config = ConfigDict(extra="forbid")

    source_account_code: str

    action: Literal["approve", "reject", "quarantine"]

    target_account_code: str | None = None

    reviewer: str = Field(min_length=1)

    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review(self) -> "HumanReviewDecision":
        if self.action == "approve":
            if not self.target_account_code:
                raise ValueError(
                    "An approval must specify a target account code."
                )

        if self.target_account_code is not None:
            self.target_account_code = self.target_account_code.strip()

        if not self.reviewer.strip():
            raise ValueError("Reviewer name cannot be blank.")

        if not self.notes.strip():
            raise ValueError("Review notes cannot be blank.")

        return self


class HumanReviewSubmission(BaseModel):
    """Collection of reviewer decisions for a pending graph run."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[HumanReviewDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "HumanReviewSubmission":
        source_codes = [
            decision.source_account_code
            for decision in self.decisions
        ]

        if len(source_codes) != len(set(source_codes)):
            raise ValueError(
                "Each source account may have only one review decision."
            )

        return self
