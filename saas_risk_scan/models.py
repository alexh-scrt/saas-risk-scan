"""Pydantic v2 data models for the SaaS Risk Scan application.

This module defines the core data contracts used throughout the application:
- SaasTool: represents a single SaaS tool input from the user
- RiskScore: holds all five risk dimension scores and computed displacement score
- AnalysisResult: combines a SaasTool with its RiskScore and metadata
- SaasStack: top-level container for a list of SaaS tools (for file input)
- RiskLevel: enum for categorized risk bands
- ReplacementTimeline: enum for estimated replacement timeline bands
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolCategory(str, Enum):
    """Supported SaaS tool categories."""

    AUTOMATION = "automation"
    CRM = "crm"
    CUSTOMER_SUPPORT = "customer_support"
    DATA_ANALYTICS = "data_analytics"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    PROJECT_MANAGEMENT = "project_management"
    COMMUNICATION = "communication"
    HR = "hr"
    FINANCE = "finance"
    MARKETING = "marketing"
    DEVTOOLS = "devtools"
    SECURITY = "security"
    STORAGE = "storage"
    ECOMMERCE = "ecommerce"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level bands derived from displacement score."""

    CRITICAL = "critical"  # 75–100
    HIGH = "high"          # 50–74
    MEDIUM = "medium"      # 25–49
    LOW = "low"            # 0–24

    @property
    def emoji(self) -> str:
        """Return an emoji indicator for the risk level."""
        mapping = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
        }
        return mapping[self]

    @property
    def label(self) -> str:
        """Return a human-readable label for the risk level."""
        return self.value.capitalize()


class ReplacementTimeline(str, Enum):
    """Estimated replacement timeline bands."""

    NEAR = "near"    # < 12 months
    MID = "mid"      # 12–24 months
    LONG = "long"    # 24–36 months
    UNLIKELY = "unlikely"  # 36+ months or very low risk

    @property
    def display(self) -> str:
        """Return a human-readable timeline description."""
        mapping = {
            ReplacementTimeline.NEAR: "< 12 months",
            ReplacementTimeline.MID: "12–24 months",
            ReplacementTimeline.LONG: "24–36 months",
            ReplacementTimeline.UNLIKELY: "36+ months",
        }
        return mapping[self]


# Score dimension type: constrained float in [0.0, 10.0]
DimensionScore = Annotated[float, Field(ge=0.0, le=10.0)]

# Overall displacement score: constrained float in [0.0, 100.0]
DisplacementScore = Annotated[float, Field(ge=0.0, le=100.0)]


class SaasTool(BaseModel):
    """Represents a single SaaS tool provided as input.

    Attributes:
        name: The name of the SaaS tool (e.g., "Zapier", "Salesforce").
        category: Tool category from the ToolCategory enum.
        monthly_cost_usd: Optional monthly spend in USD.
        team_size: Optional number of users/seats.
        notes: Optional free-text notes providing context for enrichment.
    """

    model_config = {"str_strip_whitespace": True, "frozen": False}

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the SaaS tool",
        examples=["Zapier", "Salesforce", "Notion"],
    )
    category: ToolCategory = Field(
        ...,
        description="Tool category",
        examples=["automation", "crm", "knowledge_management"],
    )
    monthly_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Monthly spend in USD (optional)",
        examples=[599.0, 1200.0],
    )
    team_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of users or seats (optional)",
        examples=[50, 10],
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Free-text notes for context (optional)",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        """Ensure the tool name is not blank after stripping whitespace."""
        if not v.strip():
            raise ValueError("Tool name must not be blank")
        return v

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: object) -> object:
        """Normalize category to lowercase string before validation."""
        if isinstance(v, str):
            return v.lower().strip()
        return v

    def display_name(self) -> str:
        """Return the display name, truncated for terminal output."""
        return self.name[:40]


class RiskScore(BaseModel):
    """Holds all five risk dimension scores and the computed displacement score.

    Each dimension is scored 0–10 (higher = greater displacement risk).
    The overall displacement_score is a weighted 0–100 composite.

    Dimension weights:
        - task_automation_ratio: 30%
        - api_openness:          20%
        - workflow_complexity:   15%  (inverted: lower complexity = higher score)
        - data_sensitivity:      20%  (inverted: higher sensitivity = lower score)
        - incumbent_inertia:     15%  (inverted: higher inertia = lower score)

    Attributes:
        task_automation_ratio: How much of the tool's value is pure task execution (0–10).
        api_openness: Quality and completeness of public APIs/webhooks (0–10).
        workflow_complexity: Depth of embedding in multi-step human workflows (0–10,
            lower complexity = higher displacement risk).
        data_sensitivity: Risk and friction of data migration / lock-in (0–10,
            higher sensitivity = lower displacement risk).
        incumbent_inertia: Organizational switching cost (0–10,
            higher inertia = lower displacement risk).
        displacement_score: Weighted composite score 0–100.
        risk_level: Categorical risk band derived from displacement_score.
        timeline: Estimated replacement timeline derived from displacement_score.
        alternatives: List of suggested agentic or open-source alternatives.
        rationale: Optional human-readable explanation of the scoring.
        enriched_by_llm: Whether this score was generated or supplemented by LLM.
    """

    model_config = {"frozen": False}

    # Five risk dimension scores (0–10)
    task_automation_ratio: DimensionScore = Field(
        ...,
        description=(
            "How much of the tool's core value is pure task execution "
            "vs. judgment (0=fully human-judgment, 10=fully automatable)"
        ),
    )
    api_openness: DimensionScore = Field(
        ...,
        description=(
            "Quality and completeness of public APIs / webhook support "
            "(0=closed/no API, 10=full open API)"
        ),
    )
    workflow_complexity: DimensionScore = Field(
        ...,
        description=(
            "How deeply the tool is embedded in multi-step human workflows "
            "(0=simple/standalone, 10=deeply complex); "
            "note: lower complexity = higher displacement risk"
        ),
    )
    data_sensitivity: DimensionScore = Field(
        ...,
        description=(
            "Risk and friction of data migration / lock-in "
            "(0=no sensitive data, 10=highly sensitive/locked in); "
            "note: higher sensitivity = lower displacement risk"
        ),
    )
    incumbent_inertia: DimensionScore = Field(
        ...,
        description=(
            "Organizational switching cost: contracts, training, political capital "
            "(0=easy to switch, 10=extremely entrenched); "
            "note: higher inertia = lower displacement risk"
        ),
    )

    # Computed composite score
    displacement_score: DisplacementScore = Field(
        ...,
        description="Weighted composite displacement risk score (0–100)",
    )

    # Derived categorical fields
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk band derived from displacement_score",
    )
    timeline: ReplacementTimeline = Field(
        ...,
        description="Estimated replacement timeline derived from displacement_score",
    )

    # Metadata
    alternatives: list[str] = Field(
        default_factory=list,
        description="Suggested agentic or open-source alternatives",
        max_length=20,
    )
    rationale: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Human-readable explanation of the scoring",
    )
    enriched_by_llm: bool = Field(
        default=False,
        description="Whether this score was generated or supplemented by LLM",
    )

    @classmethod
    def from_dimensions(
        cls,
        task_automation_ratio: float,
        api_openness: float,
        workflow_complexity: float,
        data_sensitivity: float,
        incumbent_inertia: float,
        alternatives: Optional[list[str]] = None,
        rationale: Optional[str] = None,
        enriched_by_llm: bool = False,
    ) -> "RiskScore":
        """Construct a RiskScore by computing the weighted displacement score.

        The displacement score formula:
            score = (
                task_automation_ratio * 3.0  # weight 30%
                + api_openness * 2.0          # weight 20%
                + (10 - workflow_complexity) * 1.5   # weight 15%, inverted
                + (10 - data_sensitivity) * 2.0      # weight 20%, inverted
                + (10 - incumbent_inertia) * 1.5     # weight 15%, inverted
            )
            # Normalize to 0–100 (max raw = 100 at perfect conditions)

        Args:
            task_automation_ratio: Score 0–10.
            api_openness: Score 0–10.
            workflow_complexity: Score 0–10 (inverted in weighting).
            data_sensitivity: Score 0–10 (inverted in weighting).
            incumbent_inertia: Score 0–10 (inverted in weighting).
            alternatives: Optional list of alternative tool suggestions.
            rationale: Optional explanation text.
            enriched_by_llm: Whether LLM generated this score.

        Returns:
            A fully constructed RiskScore with derived fields.
        """
        displacement_score = compute_displacement_score(
            task_automation_ratio=task_automation_ratio,
            api_openness=api_openness,
            workflow_complexity=workflow_complexity,
            data_sensitivity=data_sensitivity,
            incumbent_inertia=incumbent_inertia,
        )
        risk_level = score_to_risk_level(displacement_score)
        timeline = score_to_timeline(displacement_score)

        return cls(
            task_automation_ratio=task_automation_ratio,
            api_openness=api_openness,
            workflow_complexity=workflow_complexity,
            data_sensitivity=data_sensitivity,
            incumbent_inertia=incumbent_inertia,
            displacement_score=displacement_score,
            risk_level=risk_level,
            timeline=timeline,
            alternatives=alternatives or [],
            rationale=rationale,
            enriched_by_llm=enriched_by_llm,
        )

    def top_alternative(self) -> Optional[str]:
        """Return the first (highest priority) alternative suggestion, or None."""
        return self.alternatives[0] if self.alternatives else None

    def alternatives_display(self, max_items: int = 3) -> str:
        """Return a comma-separated string of alternatives for display purposes."""
        items = self.alternatives[:max_items]
        return ", ".join(items) if items else "—"


class AnalysisResult(BaseModel):
    """Combines a SaasTool with its computed RiskScore and analysis metadata.

    This is the primary output unit of the scoring pipeline, containing
    everything needed to render a report row.

    Attributes:
        tool: The original SaasTool input.
        score: The computed RiskScore with all dimensions and metadata.
        rank: Rank by displacement score (1 = highest risk). Set after sorting.
        source: Where the score came from: 'knowledge_base', 'llm', or 'default'.
    """

    model_config = {"frozen": False}

    tool: SaasTool = Field(..., description="The original SaaS tool input")
    score: RiskScore = Field(..., description="The computed risk score with all dimensions")
    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Rank by displacement score (1 = highest risk). Set after sorting.",
    )
    source: str = Field(
        default="default",
        description="Score source: 'knowledge_base', 'llm', or 'default'",
        pattern=r"^(knowledge_base|llm|default)$",
    )

    @property
    def tool_name(self) -> str:
        """Convenience accessor for the tool name."""
        return self.tool.name

    @property
    def category(self) -> ToolCategory:
        """Convenience accessor for the tool category."""
        return self.tool.category

    @property
    def displacement_score(self) -> float:
        """Convenience accessor for the displacement score."""
        return self.score.displacement_score

    @property
    def risk_level(self) -> RiskLevel:
        """Convenience accessor for the risk level."""
        return self.score.risk_level

    @property
    def timeline(self) -> ReplacementTimeline:
        """Convenience accessor for the replacement timeline."""
        return self.score.timeline

    def to_dict(self) -> dict[str, object]:
        """Serialize the analysis result to a plain dict for JSON export."""
        return {
            "rank": self.rank,
            "tool": {
                "name": self.tool.name,
                "category": self.tool.category.value,
                "monthly_cost_usd": self.tool.monthly_cost_usd,
                "team_size": self.tool.team_size,
                "notes": self.tool.notes,
            },
            "score": {
                "displacement_score": round(self.score.displacement_score, 1),
                "risk_level": self.score.risk_level.value,
                "timeline": self.score.timeline.value,
                "timeline_display": self.score.timeline.display,
                "dimensions": {
                    "task_automation_ratio": self.score.task_automation_ratio,
                    "api_openness": self.score.api_openness,
                    "workflow_complexity": self.score.workflow_complexity,
                    "data_sensitivity": self.score.data_sensitivity,
                    "incumbent_inertia": self.score.incumbent_inertia,
                },
                "alternatives": self.score.alternatives,
                "rationale": self.score.rationale,
                "enriched_by_llm": self.score.enriched_by_llm,
            },
            "source": self.source,
        }


class SaasStack(BaseModel):
    """Top-level container for a list of SaaS tools, used for file input parsing.

    Attributes:
        tools: List of SaasTool instances to be analyzed.
    """

    model_config = {"frozen": False}

    tools: list[SaasTool] = Field(
        ...,
        min_length=1,
        description="List of SaaS tools to analyze",
    )

    @model_validator(mode="after")
    def check_no_duplicate_names(self) -> "SaasStack":
        """Warn (but do not fail) on duplicate tool names."""
        names = [t.name.lower() for t in self.tools]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            # Store on the model for caller inspection; don't raise
            object.__setattr__(self, "_duplicate_names", duplicates)
        return self

    def duplicate_names(self) -> list[str]:
        """Return list of duplicate tool names detected during validation."""
        return getattr(self, "_duplicate_names", [])

    def tool_count(self) -> int:
        """Return the number of tools in the stack."""
        return len(self.tools)

    def total_monthly_cost(self) -> Optional[float]:
        """Return total monthly cost if any tools have cost data, else None."""
        costs = [t.monthly_cost_usd for t in self.tools if t.monthly_cost_usd is not None]
        if not costs:
            return None
        return sum(costs)


class ScanReport(BaseModel):
    """Top-level report produced after scanning an entire SaaS stack.

    Attributes:
        results: Ranked list of AnalysisResult items (highest risk first).
        generated_at: ISO 8601 timestamp of when the scan was performed.
        total_tools: Total number of tools analyzed.
        enrichment_enabled: Whether LLM enrichment was active during the scan.
        summary_stats: Aggregated statistics across all results.
    """

    model_config = {"frozen": False}

    results: list[AnalysisResult] = Field(
        default_factory=list,
        description="Ranked analysis results (highest displacement risk first)",
    )
    generated_at: str = Field(
        ...,
        description="ISO 8601 timestamp of when the scan was performed",
    )
    total_tools: int = Field(
        default=0,
        ge=0,
        description="Total number of tools analyzed",
    )
    enrichment_enabled: bool = Field(
        default=False,
        description="Whether LLM enrichment was active during this scan",
    )
    summary_stats: dict[str, object] = Field(
        default_factory=dict,
        description="Aggregated statistics (avg score, risk level counts, etc.)",
    )

    def compute_summary(self) -> None:
        """Compute and store summary statistics from the results list.

        Populates summary_stats with:
            - avg_displacement_score
            - risk_level_counts (dict of level -> count)
            - total_monthly_cost_usd
            - tools_enriched_by_llm
        """
        if not self.results:
            self.summary_stats = {
                "avg_displacement_score": 0.0,
                "risk_level_counts": {},
                "total_monthly_cost_usd": None,
                "tools_enriched_by_llm": 0,
            }
            return

        scores = [r.displacement_score for r in self.results]
        avg_score = sum(scores) / len(scores)

        risk_counts: dict[str, int] = {}
        for result in self.results:
            level = result.risk_level.value
            risk_counts[level] = risk_counts.get(level, 0) + 1

        costs = [
            r.tool.monthly_cost_usd
            for r in self.results
            if r.tool.monthly_cost_usd is not None
        ]
        total_cost: Optional[float] = sum(costs) if costs else None

        enriched_count = sum(
            1 for r in self.results if r.score.enriched_by_llm
        )

        self.summary_stats = {
            "avg_displacement_score": round(avg_score, 1),
            "risk_level_counts": risk_counts,
            "total_monthly_cost_usd": total_cost,
            "tools_enriched_by_llm": enriched_count,
        }
        self.total_tools = len(self.results)

    def rank_results(self) -> None:
        """Sort results by displacement_score descending and assign rank values."""
        self.results.sort(key=lambda r: r.displacement_score, reverse=True)
        for idx, result in enumerate(self.results, start=1):
            result.rank = idx

    def to_dict(self) -> dict[str, object]:
        """Serialize the full scan report to a plain dict for JSON export."""
        return {
            "generated_at": self.generated_at,
            "total_tools": self.total_tools,
            "enrichment_enabled": self.enrichment_enabled,
            "summary_stats": self.summary_stats,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Standalone helper functions used by RiskScore.from_dimensions
# ---------------------------------------------------------------------------


def compute_displacement_score(
    task_automation_ratio: float,
    api_openness: float,
    workflow_complexity: float,
    data_sensitivity: float,
    incumbent_inertia: float,
) -> float:
    """Compute the weighted displacement score from five dimension scores.

    Formula (max raw value = 100.0 when all pro-displacement dimensions are at
    maximum and all resistance dimensions are at minimum):

        raw = (task_automation_ratio * 3.0
               + api_openness * 2.0
               + (10 - workflow_complexity) * 1.5
               + (10 - data_sensitivity) * 2.0
               + (10 - incumbent_inertia) * 1.5)

    The raw value already has a natural maximum of 100 and minimum of 0, so no
    further normalization is needed.

    Args:
        task_automation_ratio: 0–10, higher = more automatable.
        api_openness: 0–10, higher = more open APIs.
        workflow_complexity: 0–10, higher = more complex (lowers displacement).
        data_sensitivity: 0–10, higher = more sensitive data (lowers displacement).
        incumbent_inertia: 0–10, higher = more entrenched (lowers displacement).

    Returns:
        Displacement score in [0.0, 100.0], rounded to one decimal place.
    """
    raw = (
        task_automation_ratio * 3.0
        + api_openness * 2.0
        + (10.0 - workflow_complexity) * 1.5
        + (10.0 - data_sensitivity) * 2.0
        + (10.0 - incumbent_inertia) * 1.5
    )
    # Clamp to [0, 100] to guard against float edge cases
    clamped = max(0.0, min(100.0, raw))
    return round(clamped, 1)


def score_to_risk_level(displacement_score: float) -> RiskLevel:
    """Map a displacement score to a categorical RiskLevel.

    Bands:
        75–100 → CRITICAL
        50–74  → HIGH
        25–49  → MEDIUM
        0–24   → LOW

    Args:
        displacement_score: Float in [0.0, 100.0].

    Returns:
        Corresponding RiskLevel enum value.
    """
    if displacement_score >= 75.0:
        return RiskLevel.CRITICAL
    elif displacement_score >= 50.0:
        return RiskLevel.HIGH
    elif displacement_score >= 25.0:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def score_to_timeline(displacement_score: float) -> ReplacementTimeline:
    """Map a displacement score to an estimated ReplacementTimeline.

    Bands:
        75–100 → NEAR     (< 12 months)
        50–74  → MID      (12–24 months)
        25–49  → LONG     (24–36 months)
        0–24   → UNLIKELY (36+ months)

    Args:
        displacement_score: Float in [0.0, 100.0].

    Returns:
        Corresponding ReplacementTimeline enum value.
    """
    if displacement_score >= 75.0:
        return ReplacementTimeline.NEAR
    elif displacement_score >= 50.0:
        return ReplacementTimeline.MID
    elif displacement_score >= 25.0:
        return ReplacementTimeline.LONG
    else:
        return ReplacementTimeline.UNLIKELY
