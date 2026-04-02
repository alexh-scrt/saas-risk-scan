"""Optional OpenAI-powered enrichment for unknown SaaS tools.

This module provides LLM-based scoring and alternative suggestions for tools
that are not found in the static knowledge base. It uses the OpenAI API with
the gpt-4o-mini model for cost-efficient enrichment.

Behavior:
    - If OPENAI_API_KEY is set: calls the OpenAI API to score unknown tools
      and generate agentic alternative suggestions.
    - If OPENAI_API_KEY is not set or the API call fails: gracefully falls back
      to the rule-based scorer's category defaults with a warning.
    - Known tools (in the knowledge base) are passed through unchanged and scored
      by the rule-based engine.
    - All LLM-enriched results have enriched_by_llm=True on their RiskScore.

Environment variables:
    OPENAI_API_KEY          : Required for LLM enrichment.
    SAAS_RISK_OPENAI_MODEL  : Override the default model (default: gpt-4o-mini).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from saas_risk_scan.knowledge_base import is_known
from saas_risk_scan.models import (
    AnalysisResult,
    RiskScore,
    SaasStack,
    SaasTool,
    ToolCategory,
    compute_displacement_score,
    score_to_risk_level,
    score_to_timeline,
)
from saas_risk_scan.scorer import (
    _build_dimensions_from_defaults,
    _get_default_alternatives,
    score_tool,
)

logger = logging.getLogger(__name__)

# Default OpenAI model to use for enrichment
_DEFAULT_MODEL = "gpt-4o-mini"

# Maximum number of alternatives to request from the LLM
_MAX_ALTERNATIVES = 5

# JSON schema description for the LLM response
_SCORE_SCHEMA = """
{
  "task_automation_ratio": <float 0-10>,
  "api_openness": <float 0-10>,
  "workflow_complexity": <float 0-10>,
  "data_sensitivity": <float 0-10>,
  "incumbent_inertia": <float 0-10>,
  "alternatives": [<string>, ...],
  "rationale": <string>
}
"""

_SYSTEM_PROMPT = """You are an expert technology analyst specializing in SaaS tool risk assessment 
and AI/agent displacement analysis. Your task is to score SaaS tools on five risk dimensions 
for AI displacement risk.

Scoring dimensions (each 0-10):
1. task_automation_ratio: How automatable is the tool's core value by AI agents? 
   (0=requires full human judgment, 10=fully automatable by agents)
2. api_openness: Quality and completeness of public APIs/webhooks 
   (0=no API/closed, 10=excellent open API with full functionality)
3. workflow_complexity: How deeply embedded in multi-step human workflows? 
   (0=simple standalone, 10=deeply complex multi-system workflows)
   Note: LOWER complexity means HIGHER displacement risk
4. data_sensitivity: Risk and friction of data migration/lock-in 
   (0=no sensitive data, 10=highly sensitive/regulated data)
   Note: HIGHER sensitivity means LOWER displacement risk
5. incumbent_inertia: Organizational switching cost 
   (0=easy to switch, 10=extremely entrenched)
   Note: HIGHER inertia means LOWER displacement risk

Also provide:
- alternatives: List of 3-5 concrete agentic, open-source, or AI-native replacement suggestions
- rationale: 2-4 sentence explanation of the scores

Respond ONLY with valid JSON matching the schema provided. No markdown, no explanation outside JSON."""


def _get_openai_client() -> Optional[object]:
    """Attempt to create and return an OpenAI client.

    Returns:
        An OpenAI client instance if the API key is available, or None.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore[import]
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai package not available; LLM enrichment disabled.")
        return None
    except Exception as exc:
        logger.warning("Failed to initialize OpenAI client: %s", exc)
        return None


def _get_model() -> str:
    """Return the OpenAI model to use, from environment or default.

    Returns:
        Model name string.
    """
    return os.environ.get("SAAS_RISK_OPENAI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _build_user_prompt(tool: SaasTool) -> str:
    """Build the user prompt for a single SaaS tool scoring request.

    Args:
        tool: The SaasTool to score.

    Returns:
        Formatted prompt string.
    """
    lines = [
        f"Please score the following SaaS tool for AI displacement risk:",
        f"",
        f"Tool Name: {tool.name}",
        f"Category: {tool.category.value}",
    ]
    if tool.monthly_cost_usd is not None:
        lines.append(f"Monthly Cost: ${tool.monthly_cost_usd:.0f}/month")
    if tool.team_size is not None:
        lines.append(f"Team Size: {tool.team_size} users")
    if tool.notes:
        lines.append(f"Context: {tool.notes}")
    lines.extend([
        f"",
        f"Respond with JSON matching this exact schema:",
        _SCORE_SCHEMA,
    ])
    return "\n".join(lines)


def _parse_llm_response(response_text: str, tool_name: str) -> Optional[dict[str, object]]:
    """Parse and validate the JSON response from the LLM.

    Args:
        response_text: Raw text response from the LLM.
        tool_name: Tool name for error context.

    Returns:
        Parsed dictionary if valid, or None on parse/validation failure.
    """
    # Strip any markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response for '%s': %s", tool_name, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("LLM response for '%s' is not a JSON object", tool_name)
        return None

    required_dimensions = {
        "task_automation_ratio",
        "api_openness",
        "workflow_complexity",
        "data_sensitivity",
        "incumbent_inertia",
    }
    missing = required_dimensions - set(data.keys())
    if missing:
        logger.warning(
            "LLM response for '%s' missing dimension keys: %s", tool_name, missing
        )
        return None

    # Validate and clamp dimension scores
    for dim in required_dimensions:
        try:
            val = float(data[dim])  # type: ignore[arg-type]
            data[dim] = max(0.0, min(10.0, val))
        except (TypeError, ValueError):
            logger.warning(
                "LLM response for '%s' has invalid value for '%s': %r",
                tool_name, dim, data[dim],
            )
            return None

    # Validate alternatives
    if "alternatives" not in data or not isinstance(data["alternatives"], list):
        data["alternatives"] = []
    else:
        # Keep only string alternatives, max _MAX_ALTERNATIVES
        data["alternatives"] = [
            str(a) for a in data["alternatives"] if a
        ][:_MAX_ALTERNATIVES]

    # Validate rationale
    if "rationale" not in data or not isinstance(data["rationale"], str):
        data["rationale"] = f"LLM-generated risk assessment for {tool_name}."
    else:
        data["rationale"] = str(data["rationale"]).strip()

    return data


def _enrich_single_tool(
    tool: SaasTool,
    client: object,
    model: str,
) -> Optional[AnalysisResult]:
    """Attempt to enrich a single tool using the OpenAI API.

    Args:
        tool: The SaasTool to enrich.
        client: An initialized OpenAI client.
        model: The model name to use.

    Returns:
        An AnalysisResult with LLM-generated scores if successful, or None
        if the API call fails or the response cannot be parsed.
    """
    try:
        from openai import OpenAI  # type: ignore[import]
        from openai import APIError, APITimeoutError, RateLimitError  # type: ignore[import]
    except ImportError:
        return None

    user_prompt = _build_user_prompt(tool)

    try:
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"} if "gpt-4" in model or "gpt-3.5" in model else None,
        )
    except Exception as exc:
        logger.warning("OpenAI API call failed for '%s': %s", tool.name, exc)
        return None

    if not response.choices:
        logger.warning("OpenAI returned no choices for '%s'", tool.name)
        return None

    content = response.choices[0].message.content or ""
    parsed = _parse_llm_response(content, tool.name)
    if parsed is None:
        return None

    displacement_score = compute_displacement_score(
        task_automation_ratio=float(parsed["task_automation_ratio"]),  # type: ignore[arg-type]
        api_openness=float(parsed["api_openness"]),  # type: ignore[arg-type]
        workflow_complexity=float(parsed["workflow_complexity"]),  # type: ignore[arg-type]
        data_sensitivity=float(parsed["data_sensitivity"]),  # type: ignore[arg-type]
        incumbent_inertia=float(parsed["incumbent_inertia"]),  # type: ignore[arg-type]
    )

    risk_score = RiskScore.from_dimensions(
        task_automation_ratio=float(parsed["task_automation_ratio"]),  # type: ignore[arg-type]
        api_openness=float(parsed["api_openness"]),  # type: ignore[arg-type]
        workflow_complexity=float(parsed["workflow_complexity"]),  # type: ignore[arg-type]
        data_sensitivity=float(parsed["data_sensitivity"]),  # type: ignore[arg-type]
        incumbent_inertia=float(parsed["incumbent_inertia"]),  # type: ignore[arg-type]
        alternatives=list(parsed["alternatives"]),  # type: ignore[arg-type]
        rationale=str(parsed["rationale"]),
        enriched_by_llm=True,
    )

    return AnalysisResult(
        tool=tool,
        score=risk_score,
        source="llm",
    )


def is_enrichment_available() -> bool:
    """Check whether LLM enrichment is available (API key is set).

    Returns:
        True if OPENAI_API_KEY is set and non-empty, False otherwise.
    """
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def enrich_tool(
    tool: SaasTool,
    fallback_to_defaults: bool = True,
) -> AnalysisResult:
    """Enrich a single SaaS tool using LLM, with graceful fallback.

    If the tool is in the knowledge base, it is scored using the rule-based
    engine (KB path) regardless of whether enrichment is requested. Only tools
    NOT in the knowledge base are sent to the LLM.

    If the LLM is unavailable or fails, falls back to the rule-based scorer's
    category defaults (if fallback_to_defaults=True).

    Args:
        tool: The SaasTool to enrich.
        fallback_to_defaults: If True, fall back to rule-based scoring on LLM
            failure. If False, raises an exception on failure.

    Returns:
        An AnalysisResult with LLM or fallback scores.

    Raises:
        RuntimeError: If fallback_to_defaults=False and enrichment fails.
    """
    # Known tools don't need LLM enrichment
    if is_known(tool.name):
        return score_tool(tool, use_knowledge_base=True)

    client = _get_openai_client()
    if client is None:
        if not fallback_to_defaults:
            raise RuntimeError(
                f"LLM enrichment unavailable for '{tool.name}': OPENAI_API_KEY not set"
            )
        logger.info(
            "LLM enrichment unavailable for '%s'; using rule-based defaults.", tool.name
        )
        return score_tool(tool, use_knowledge_base=False)

    model = _get_model()
    logger.info("Enriching '%s' via OpenAI (%s)...", tool.name, model)

    result = _enrich_single_tool(tool, client, model)
    if result is not None:
        return result

    # Fallback on failure
    if not fallback_to_defaults:
        raise RuntimeError(
            f"LLM enrichment failed for '{tool.name}' and fallback is disabled."
        )
    logger.warning(
        "LLM enrichment failed for '%s'; falling back to rule-based defaults.", tool.name
    )
    return score_tool(tool, use_knowledge_base=False)


def enrich_stack(
    stack: SaasStack,
    enrich_unknown_only: bool = True,
    progress_callback: Optional[object] = None,
) -> list[AnalysisResult]:
    """Enrich all tools in a SaasStack using LLM where applicable.

    For each tool in the stack:
    - If the tool is in the knowledge base (and enrich_unknown_only=True): score
      using the rule-based engine (fast, no API cost).
    - If the tool is unknown: attempt LLM enrichment with fallback to defaults.

    Args:
        stack: The SaasStack to process.
        enrich_unknown_only: If True (default), only enrich tools not in the KB.
            If False, enrich all tools via LLM.
        progress_callback: Optional callable(tool_name: str, index: int, total: int)
            called after each tool is processed (for progress reporting).

    Returns:
        List of AnalysisResult items in the same order as the input stack.
    """
    results: list[AnalysisResult] = []
    total = len(stack.tools)

    for idx, tool in enumerate(stack.tools, start=1):
        if enrich_unknown_only and is_known(tool.name):
            result = score_tool(tool, use_knowledge_base=True)
        else:
            result = enrich_tool(tool, fallback_to_defaults=True)

        results.append(result)

        if progress_callback is not None:
            try:
                progress_callback(tool.name, idx, total)  # type: ignore[operator]
            except Exception:
                pass  # Never let a callback break the main flow

    return results


def get_enrichment_summary(results: list[AnalysisResult]) -> dict[str, int]:
    """Count how many results came from each source in a list of AnalysisResults.

    Args:
        results: List of AnalysisResult items.

    Returns:
        Dictionary with keys 'knowledge_base', 'llm', 'default' mapping to counts.
    """
    summary: dict[str, int] = {"knowledge_base": 0, "llm": 0, "default": 0}
    for result in results:
        source = result.source
        if source in summary:
            summary[source] += 1
        else:
            summary[source] = 1
    return summary
