"""Core rule-based scoring engine for the SaaS Risk Scan tool.

This module provides the primary scoring pipeline that computes all five risk
dimension scores and produces a weighted 0–100 displacement score with estimated
replacement timelines for each SaaS tool in a stack.

Scoring strategy:
    1. If the tool is in the knowledge base, use the pre-researched baseline
       dimension scores directly.
    2. If the tool is unknown, apply heuristic rules based on available metadata
       (category, team size, cost, notes keywords) to estimate dimension scores.
    3. Adjust baseline scores using optional contextual modifiers derived from
       the SaasTool metadata (cost signals, team size signals, notes keywords).
    4. Compute the weighted displacement score and derive risk level / timeline.

The five dimensions and their weights:
    - task_automation_ratio : 30% (higher = more automatable by agents)
    - api_openness          : 20% (higher = better public API)
    - workflow_complexity   : 15% (lower complexity = higher displacement risk)
    - data_sensitivity      : 20% (higher sensitivity = lower displacement risk)
    - incumbent_inertia     : 15% (higher inertia = lower displacement risk)
"""

from __future__ import annotations

import re
from typing import Optional

from saas_risk_scan.knowledge_base import KnowledgeBaseEntry, lookup
from saas_risk_scan.models import (
    AnalysisResult,
    RiskScore,
    SaasStack,
    SaasTool,
    ScanReport,
    ToolCategory,
    compute_displacement_score,
    score_to_risk_level,
    score_to_timeline,
)

# ---------------------------------------------------------------------------
# Category-level default dimension scores for unknown tools
# Used when a tool is not in the knowledge base.
# Scores represent the "typical" tool in that category.
# ---------------------------------------------------------------------------

_CATEGORY_DEFAULTS: dict[ToolCategory, dict[str, float]] = {
    ToolCategory.AUTOMATION: {
        "task_automation_ratio": 8.5,
        "api_openness": 8.0,
        "workflow_complexity": 4.0,
        "data_sensitivity": 2.5,
        "incumbent_inertia": 4.0,
    },
    ToolCategory.CRM: {
        "task_automation_ratio": 5.5,
        "api_openness": 7.0,
        "workflow_complexity": 6.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 6.0,
    },
    ToolCategory.CUSTOMER_SUPPORT: {
        "task_automation_ratio": 7.5,
        "api_openness": 7.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.5,
    },
    ToolCategory.DATA_ANALYTICS: {
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 6.0,
    },
    ToolCategory.KNOWLEDGE_MANAGEMENT: {
        "task_automation_ratio": 5.0,
        "api_openness": 6.5,
        "workflow_complexity": 6.0,
        "data_sensitivity": 4.5,
        "incumbent_inertia": 5.5,
    },
    ToolCategory.PROJECT_MANAGEMENT: {
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 6.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 5.5,
    },
    ToolCategory.COMMUNICATION: {
        "task_automation_ratio": 4.0,
        "api_openness": 7.5,
        "workflow_complexity": 7.0,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 7.0,
    },
    ToolCategory.HR: {
        "task_automation_ratio": 4.5,
        "api_openness": 6.0,
        "workflow_complexity": 7.0,
        "data_sensitivity": 8.5,
        "incumbent_inertia": 7.0,
    },
    ToolCategory.FINANCE: {
        "task_automation_ratio": 5.0,
        "api_openness": 6.5,
        "workflow_complexity": 6.0,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 7.0,
    },
    ToolCategory.MARKETING: {
        "task_automation_ratio": 7.0,
        "api_openness": 7.5,
        "workflow_complexity": 5.0,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.0,
    },
    ToolCategory.DEVTOOLS: {
        "task_automation_ratio": 5.0,
        "api_openness": 8.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 6.5,
    },
    ToolCategory.SECURITY: {
        "task_automation_ratio": 3.5,
        "api_openness": 7.0,
        "workflow_complexity": 7.0,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 8.0,
    },
    ToolCategory.STORAGE: {
        "task_automation_ratio": 4.0,
        "api_openness": 7.5,
        "workflow_complexity": 5.0,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 5.5,
    },
    ToolCategory.ECOMMERCE: {
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 6.5,
    },
    ToolCategory.OTHER: {
        "task_automation_ratio": 5.0,
        "api_openness": 6.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.0,
    },
}

# Fallback defaults when category is not mapped (should not happen with current enum)
_GLOBAL_DEFAULT: dict[str, float] = {
    "task_automation_ratio": 5.0,
    "api_openness": 6.5,
    "workflow_complexity": 5.5,
    "data_sensitivity": 5.0,
    "incumbent_inertia": 5.0,
}

# ---------------------------------------------------------------------------
# Category-level default alternatives for unknown tools
# ---------------------------------------------------------------------------

_CATEGORY_DEFAULT_ALTERNATIVES: dict[ToolCategory, list[str]] = {
    ToolCategory.AUTOMATION: ["n8n", "LangChain agents", "AutoGen", "Prefect"],
    ToolCategory.CRM: ["Twenty CRM", "Attio", "custom LLM CRM pipeline"],
    ToolCategory.CUSTOMER_SUPPORT: ["Chatwoot", "Botpress", "custom LLM support agent"],
    ToolCategory.DATA_ANALYTICS: ["Metabase", "Apache Superset", "AI SQL agents"],
    ToolCategory.KNOWLEDGE_MANAGEMENT: ["Outline", "Obsidian + AI", "custom RAG pipeline"],
    ToolCategory.PROJECT_MANAGEMENT: ["Plane", "Linear", "GitHub Issues + AI"],
    ToolCategory.COMMUNICATION: ["Mattermost", "Matrix/Element", "Zulip"],
    ToolCategory.HR: ["Rippling", "BambooHR", "HiBob"],
    ToolCategory.FINANCE: ["Wave", "AI bookkeeping agents", "custom finance pipeline"],
    ToolCategory.MARKETING: ["Mautic", "Listmonk", "LangChain email agents"],
    ToolCategory.DEVTOOLS: ["open-source equivalent", "self-hosted stack", "custom agent tooling"],
    ToolCategory.SECURITY: ["Keycloak", "open-source IAM", "Authentik"],
    ToolCategory.STORAGE: ["Nextcloud", "self-hosted S3", "Seafile"],
    ToolCategory.ECOMMERCE: ["Medusa.js", "WooCommerce", "Saleor"],
    ToolCategory.OTHER: ["open-source alternative", "custom agent solution"],
}

# ---------------------------------------------------------------------------
# Notes keyword patterns for contextual score adjustments
# Each tuple: (regex pattern, dimension, delta)
# Delta is applied to the named dimension score (clamped to [0, 10]).
# ---------------------------------------------------------------------------

_NOTES_ADJUSTMENTS: list[tuple[re.Pattern[str], str, float]] = [
    # High automation signals → raise task_automation_ratio
    (re.compile(r"\bautomat", re.IGNORECASE), "task_automation_ratio", +0.5),
    (re.compile(r"\btrigger", re.IGNORECASE), "task_automation_ratio", +0.3),
    (re.compile(r"\bzap|\.workflow|\.automation", re.IGNORECASE), "task_automation_ratio", +0.3),
    # Compliance / regulatory mentions → raise data_sensitivity
    (re.compile(r"\bcomplian", re.IGNORECASE), "data_sensitivity", +0.5),
    (re.compile(r"\baudit|\bregulat", re.IGNORECASE), "data_sensitivity", +0.5),
    (re.compile(r"\bpayroll|\bpii|\bpci", re.IGNORECASE), "data_sensitivity", +1.0),
    # Deep integration / customization → raise workflow_complexity and inertia
    (re.compile(r"\bcustom|\bheavily|\bcomplex", re.IGNORECASE), "workflow_complexity", +0.5),
    (re.compile(r"\bcustom|\bheavily|\bcomplex", re.IGNORECASE), "incumbent_inertia", +0.5),
    (re.compile(r"\bintegrat", re.IGNORECASE), "workflow_complexity", +0.3),
    # API / open mentions → raise api_openness
    (re.compile(r"\bapi|\bwebhook|\bopen.source", re.IGNORECASE), "api_openness", +0.5),
    # Years of data / history → raise data_sensitivity and inertia
    (re.compile(r"\b\d+\s+years?\b", re.IGNORECASE), "data_sensitivity", +0.5),
    (re.compile(r"\b\d+\s+years?\b", re.IGNORECASE), "incumbent_inertia", +0.5),
    # Contract / enterprise mentions → raise incumbent_inertia
    (re.compile(r"\bcontract|\benterprise", re.IGNORECASE), "incumbent_inertia", +0.5),
    # Migration underway → lower inertia
    (re.compile(r"\bmigrat", re.IGNORECASE), "incumbent_inertia", -0.5),
    # Simple / lightweight → lower workflow_complexity
    (re.compile(r"\bsimple|\blightweight|\bbasic", re.IGNORECASE), "workflow_complexity", -0.5),
    # Security sensitive → raise data_sensitivity
    (re.compile(r"\bsensitive|\bsecure|\bencrypt", re.IGNORECASE), "data_sensitivity", +0.5),
]

# ---------------------------------------------------------------------------
# Team-size modifiers
# Large teams using a tool increases incumbent inertia.
# ---------------------------------------------------------------------------

_TEAM_SIZE_THRESHOLDS: list[tuple[int, float]] = [
    # (min_team_size, inertia_delta)
    (100, +1.5),
    (50, +1.0),
    (20, +0.5),
    (10, +0.25),
    (5, 0.0),
    (0, -0.25),  # very small team = lower inertia
]

# ---------------------------------------------------------------------------
# Cost modifiers
# High monthly spend signals deep organizational commitment → raise inertia.
# ---------------------------------------------------------------------------

_COST_THRESHOLDS: list[tuple[float, float]] = [
    # (min_monthly_cost_usd, inertia_delta)
    (5000.0, +1.5),
    (2000.0, +1.0),
    (1000.0, +0.5),
    (500.0, +0.25),
    (100.0, 0.0),
    (0.0, -0.25),  # free tool = lower inertia
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """Clamp a float value to the range [lo, hi].

    Args:
        value: The value to clamp.
        lo: Minimum bound (default 0.0).
        hi: Maximum bound (default 10.0).

    Returns:
        The clamped value.
    """
    return max(lo, min(hi, value))


def _get_category_defaults(category: ToolCategory) -> dict[str, float]:
    """Return a copy of the category-level default dimension scores.

    Args:
        category: The ToolCategory to look up.

    Returns:
        Dictionary mapping dimension name to float score (0–10).
    """
    defaults = _CATEGORY_DEFAULTS.get(category)
    if defaults is None:
        return dict(_GLOBAL_DEFAULT)
    return dict(defaults)


def _apply_team_size_modifier(
    dims: dict[str, float],
    team_size: Optional[int],
) -> dict[str, float]:
    """Apply incumbent inertia modifier based on team size.

    Larger teams adopting a tool create higher organizational inertia.

    Args:
        dims: Mutable dimension score dictionary to modify in place.
        team_size: Number of users/seats, or None if not provided.

    Returns:
        The modified dims dictionary.
    """
    if team_size is None:
        return dims

    delta = 0.0
    for threshold, d in _TEAM_SIZE_THRESHOLDS:
        if team_size >= threshold:
            delta = d
            break

    dims["incumbent_inertia"] = _clamp(dims["incumbent_inertia"] + delta)
    return dims


def _apply_cost_modifier(
    dims: dict[str, float],
    monthly_cost_usd: Optional[float],
) -> dict[str, float]:
    """Apply incumbent inertia modifier based on monthly spend.

    Higher spend implies deeper financial and organizational commitment.

    Args:
        dims: Mutable dimension score dictionary to modify in place.
        monthly_cost_usd: Monthly cost in USD, or None if not provided.

    Returns:
        The modified dims dictionary.
    """
    if monthly_cost_usd is None:
        return dims

    delta = 0.0
    for threshold, d in _COST_THRESHOLDS:
        if monthly_cost_usd >= threshold:
            delta = d
            break

    dims["incumbent_inertia"] = _clamp(dims["incumbent_inertia"] + delta)
    # Very high spend also slightly increases data_sensitivity (suggests deep usage)
    if monthly_cost_usd >= 2000.0:
        dims["data_sensitivity"] = _clamp(dims["data_sensitivity"] + 0.25)
    return dims


def _apply_notes_modifiers(
    dims: dict[str, float],
    notes: Optional[str],
) -> dict[str, float]:
    """Apply dimension score adjustments based on keyword patterns in notes.

    Scans the notes field for contextual keywords that signal higher or lower
    risk in specific dimensions.

    Args:
        dims: Mutable dimension score dictionary to modify in place.
        notes: Free-text notes from the SaasTool input, or None.

    Returns:
        The modified dims dictionary.
    """
    if not notes or not notes.strip():
        return dims

    for pattern, dimension, delta in _NOTES_ADJUSTMENTS:
        if pattern.search(notes):
            dims[dimension] = _clamp(dims[dimension] + delta)

    return dims


def _build_dimensions_from_kb(
    entry: KnowledgeBaseEntry,
    tool: SaasTool,
) -> dict[str, float]:
    """Build dimension scores starting from a knowledge base entry, then apply
    contextual modifiers from tool metadata.

    Args:
        entry: The matched KnowledgeBaseEntry with pre-researched baseline scores.
        tool: The SaasTool with optional metadata (cost, team size, notes).

    Returns:
        Dictionary of adjusted dimension scores (0–10).
    """
    dims = {
        "task_automation_ratio": entry.task_automation_ratio,
        "api_openness": entry.api_openness,
        "workflow_complexity": entry.workflow_complexity,
        "data_sensitivity": entry.data_sensitivity,
        "incumbent_inertia": entry.incumbent_inertia,
    }
    dims = _apply_team_size_modifier(dims, tool.team_size)
    dims = _apply_cost_modifier(dims, tool.monthly_cost_usd)
    dims = _apply_notes_modifiers(dims, tool.notes)
    return dims


def _build_dimensions_from_defaults(
    tool: SaasTool,
) -> dict[str, float]:
    """Build dimension scores from category-level defaults, then apply
    contextual modifiers from tool metadata.

    Used for tools that are not found in the knowledge base.

    Args:
        tool: The SaasTool with category and optional metadata.

    Returns:
        Dictionary of estimated dimension scores (0–10).
    """
    dims = _get_category_defaults(tool.category)
    dims = _apply_team_size_modifier(dims, tool.team_size)
    dims = _apply_cost_modifier(dims, tool.monthly_cost_usd)
    dims = _apply_notes_modifiers(dims, tool.notes)
    return dims


def _get_default_alternatives(category: ToolCategory) -> list[str]:
    """Return the default alternative suggestions for a given category.

    Args:
        category: The ToolCategory to look up.

    Returns:
        List of alternative tool/framework suggestion strings.
    """
    return list(_CATEGORY_DEFAULT_ALTERNATIVES.get(category, ["open-source alternative"]))


def _build_rationale_for_default(
    tool: SaasTool,
    dims: dict[str, float],
    displacement_score: float,
) -> str:
    """Generate a human-readable rationale string for an unknown tool scored
    using category defaults.

    Args:
        tool: The SaasTool input.
        dims: The computed dimension scores.
        displacement_score: The final weighted displacement score.

    Returns:
        A multi-sentence rationale string.
    """
    risk_level = score_to_risk_level(displacement_score)
    category_label = tool.category.value.replace("_", " ")
    parts = [
        f"{tool.name} is an unknown tool in the '{category_label}' category, "
        f"scored using category-level heuristic defaults.",
        f"Overall displacement risk is {risk_level.label} "
        f"(score: {displacement_score:.1f}/100).",
    ]

    # Add contextual notes
    if dims["task_automation_ratio"] >= 7.0:
        parts.append(
            "The category suggests a high degree of task automation, making it "
            "a likely target for agent-based replacement."
        )
    if dims["data_sensitivity"] >= 8.0:
        parts.append(
            "High data sensitivity and compliance requirements reduce near-term "
            "displacement risk."
        )
    if dims["incumbent_inertia"] >= 7.0:
        parts.append(
            "Strong organizational inertia is estimated, slowing replacement timelines."
        )
    if tool.notes:
        parts.append("Notes provided were used to refine dimension scores.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def score_tool(
    tool: SaasTool,
    use_knowledge_base: bool = True,
) -> AnalysisResult:
    """Score a single SaasTool using the rule-based scoring engine.

    Scoring pipeline:
    1. Check the knowledge base for a pre-researched entry.
    2. If found: use the KB baseline scores and apply metadata modifiers.
    3. If not found: use category-level defaults and apply metadata modifiers.
    4. Compute the weighted displacement score.
    5. Derive risk level and timeline.
    6. Return a fully populated AnalysisResult.

    Args:
        tool: The SaasTool to score.
        use_knowledge_base: If True (default), attempt KB lookup before falling
            back to defaults. Set to False to always use defaults (useful for
            testing or when KB lookup has already been attempted).

    Returns:
        An AnalysisResult with the computed RiskScore and source metadata.
    """
    source: str
    alternatives: list[str]
    rationale: Optional[str]

    kb_entry = lookup(tool.name) if use_knowledge_base else None

    if kb_entry is not None:
        # Path 1: Knowledge base hit
        dims = _build_dimensions_from_kb(kb_entry, tool)
        alternatives = list(kb_entry.alternatives)
        rationale = kb_entry.rationale
        source = "knowledge_base"
    else:
        # Path 2: Unknown tool — use category defaults
        dims = _build_dimensions_from_defaults(tool)
        alternatives = _get_default_alternatives(tool.category)
        # Rationale computed after score is available; placeholder until below
        rationale = None
        source = "default"

    displacement_score = compute_displacement_score(
        task_automation_ratio=dims["task_automation_ratio"],
        api_openness=dims["api_openness"],
        workflow_complexity=dims["workflow_complexity"],
        data_sensitivity=dims["data_sensitivity"],
        incumbent_inertia=dims["incumbent_inertia"],
    )

    # Generate default rationale now that we have the final score
    if rationale is None:
        rationale = _build_rationale_for_default(tool, dims, displacement_score)

    risk_score = RiskScore.from_dimensions(
        task_automation_ratio=dims["task_automation_ratio"],
        api_openness=dims["api_openness"],
        workflow_complexity=dims["workflow_complexity"],
        data_sensitivity=dims["data_sensitivity"],
        incumbent_inertia=dims["incumbent_inertia"],
        alternatives=alternatives,
        rationale=rationale,
        enriched_by_llm=False,
    )

    return AnalysisResult(
        tool=tool,
        score=risk_score,
        source=source,
    )


def score_stack(
    stack: SaasStack,
    use_knowledge_base: bool = True,
) -> list[AnalysisResult]:
    """Score all tools in a SaasStack and return a list of AnalysisResult items.

    Results are returned in the same order as the input stack. No ranking or
    sorting is applied here; use build_report() for ranked output.

    Args:
        stack: A validated SaasStack containing one or more SaasTool entries.
        use_knowledge_base: Whether to use the knowledge base for known tools.

    Returns:
        List of AnalysisResult items, one per tool in the stack.
    """
    results: list[AnalysisResult] = []
    for tool in stack.tools:
        result = score_tool(tool, use_knowledge_base=use_knowledge_base)
        results.append(result)
    return results


def build_report(
    stack: SaasStack,
    generated_at: str,
    enrichment_enabled: bool = False,
    pre_scored_results: Optional[list[AnalysisResult]] = None,
) -> ScanReport:
    """Build a complete ScanReport from a SaasStack.

    This is the top-level function that orchestrates the full scoring pipeline:
    1. Score each tool (using KB or defaults).
    2. Rank results by displacement score (highest first).
    3. Compute summary statistics.
    4. Return a ScanReport ready for rendering.

    If ``pre_scored_results`` is provided (e.g., from LLM enrichment that has
    already scored some tools), those results are used directly and merged with
    any remaining tools that still need scoring.

    Args:
        stack: The validated SaasStack to analyze.
        generated_at: ISO 8601 timestamp string for the report header.
        enrichment_enabled: Whether LLM enrichment was active during this scan.
        pre_scored_results: Optional list of AnalysisResult items that have
            already been scored (e.g., by the LLM enricher). Any tools in the
            stack not represented in this list will be scored by the rule engine.

    Returns:
        A fully populated ScanReport with ranked results and summary statistics.
    """
    if pre_scored_results is not None:
        # Build a lookup of already-scored tool names (case-insensitive)
        scored_names: set[str] = {
            r.tool.name.lower() for r in pre_scored_results
        }
        # Score any remaining tools not already covered
        remaining_results: list[AnalysisResult] = []
        for tool in stack.tools:
            if tool.name.lower() not in scored_names:
                remaining_results.append(
                    score_tool(tool, use_knowledge_base=True)
                )
        all_results = list(pre_scored_results) + remaining_results
    else:
        all_results = score_stack(stack, use_knowledge_base=True)

    report = ScanReport(
        results=all_results,
        generated_at=generated_at,
        total_tools=len(all_results),
        enrichment_enabled=enrichment_enabled,
    )
    report.rank_results()
    report.compute_summary()
    return report


def get_dimension_weights() -> dict[str, float]:
    """Return the weight assigned to each risk dimension in the scoring formula.

    These weights are for informational/display purposes. The actual computation
    is performed by ``compute_displacement_score()`` in models.py.

    Returns:
        Dictionary mapping dimension name to weight percentage (0–100 scale).
    """
    return {
        "task_automation_ratio": 30.0,
        "api_openness": 20.0,
        "workflow_complexity": 15.0,  # inverted: lower = higher displacement
        "data_sensitivity": 20.0,     # inverted: higher = lower displacement
        "incumbent_inertia": 15.0,    # inverted: higher = lower displacement
    }


def score_dimensions_only(
    task_automation_ratio: float,
    api_openness: float,
    workflow_complexity: float,
    data_sensitivity: float,
    incumbent_inertia: float,
) -> dict[str, object]:
    """Compute a displacement score from explicit dimension values and return
    a summary dictionary. Useful for testing and interactive scoring.

    Args:
        task_automation_ratio: Score 0–10.
        api_openness: Score 0–10.
        workflow_complexity: Score 0–10.
        data_sensitivity: Score 0–10.
        incumbent_inertia: Score 0–10.

    Returns:
        Dictionary with keys:
            - task_automation_ratio, api_openness, workflow_complexity,
              data_sensitivity, incumbent_inertia (float)
            - displacement_score (float)
            - risk_level (str)
            - timeline (str)
            - timeline_display (str)
    """
    score = compute_displacement_score(
        task_automation_ratio=task_automation_ratio,
        api_openness=api_openness,
        workflow_complexity=workflow_complexity,
        data_sensitivity=data_sensitivity,
        incumbent_inertia=incumbent_inertia,
    )
    risk_level = score_to_risk_level(score)
    timeline = score_to_timeline(score)
    return {
        "task_automation_ratio": task_automation_ratio,
        "api_openness": api_openness,
        "workflow_complexity": workflow_complexity,
        "data_sensitivity": data_sensitivity,
        "incumbent_inertia": incumbent_inertia,
        "displacement_score": score,
        "risk_level": risk_level.value,
        "timeline": timeline.value,
        "timeline_display": timeline.display,
    }
