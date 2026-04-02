"""Unit tests for saas_risk_scan/scorer.py.

Covers edge cases, dimension calculations, KB vs default scoring paths,
metadata modifiers, and the full report-building pipeline.
"""

from __future__ import annotations

import pytest

from saas_risk_scan.models import (
    AnalysisResult,
    ReplacementTimeline,
    RiskLevel,
    SaasStack,
    SaasTool,
    ScanReport,
    ToolCategory,
)
from saas_risk_scan.scorer import (
    _apply_cost_modifier,
    _apply_notes_modifiers,
    _apply_team_size_modifier,
    _build_dimensions_from_defaults,
    _build_dimensions_from_kb,
    _clamp,
    _get_category_defaults,
    _get_default_alternatives,
    build_report,
    get_dimension_weights,
    score_dimensions_only,
    score_stack,
    score_tool,
)
from saas_risk_scan.knowledge_base import lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool(
    name: str = "TestTool",
    category: ToolCategory = ToolCategory.OTHER,
    monthly_cost_usd: float | None = None,
    team_size: int | None = None,
    notes: str | None = None,
) -> SaasTool:
    """Convenience factory for SaasTool instances in tests."""
    return SaasTool(
        name=name,
        category=category,
        monthly_cost_usd=monthly_cost_usd,
        team_size=team_size,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    """Tests for the internal _clamp helper."""

    def test_clamp_within_range(self) -> None:
        assert _clamp(5.0) == 5.0

    def test_clamp_at_min(self) -> None:
        assert _clamp(0.0) == 0.0

    def test_clamp_at_max(self) -> None:
        assert _clamp(10.0) == 10.0

    def test_clamp_below_min(self) -> None:
        assert _clamp(-1.0) == 0.0

    def test_clamp_above_max(self) -> None:
        assert _clamp(11.0) == 10.0

    def test_clamp_custom_bounds(self) -> None:
        assert _clamp(50.0, lo=0.0, hi=100.0) == 50.0
        assert _clamp(150.0, lo=0.0, hi=100.0) == 100.0
        assert _clamp(-5.0, lo=0.0, hi=100.0) == 0.0


# ---------------------------------------------------------------------------
# _get_category_defaults
# ---------------------------------------------------------------------------


class TestGetCategoryDefaults:
    """Tests for the _get_category_defaults helper."""

    def test_returns_dict_with_five_keys(self) -> None:
        defaults = _get_category_defaults(ToolCategory.AUTOMATION)
        expected_keys = {
            "task_automation_ratio",
            "api_openness",
            "workflow_complexity",
            "data_sensitivity",
            "incumbent_inertia",
        }
        assert set(defaults.keys()) == expected_keys

    def test_all_values_in_range(self) -> None:
        for category in ToolCategory:
            defaults = _get_category_defaults(category)
            for key, val in defaults.items():
                assert 0.0 <= val <= 10.0, (
                    f"Category {category} key '{key}' out of range: {val}"
                )

    def test_returns_copy(self) -> None:
        """Modifying the returned dict should not affect future calls."""
        defaults1 = _get_category_defaults(ToolCategory.AUTOMATION)
        defaults1["task_automation_ratio"] = 0.0
        defaults2 = _get_category_defaults(ToolCategory.AUTOMATION)
        assert defaults2["task_automation_ratio"] != 0.0

    def test_automation_has_high_task_ratio(self) -> None:
        defaults = _get_category_defaults(ToolCategory.AUTOMATION)
        assert defaults["task_automation_ratio"] >= 7.0

    def test_finance_has_high_data_sensitivity(self) -> None:
        defaults = _get_category_defaults(ToolCategory.FINANCE)
        assert defaults["data_sensitivity"] >= 8.0

    def test_security_has_high_data_sensitivity(self) -> None:
        defaults = _get_category_defaults(ToolCategory.SECURITY)
        assert defaults["data_sensitivity"] >= 8.0

    def test_all_categories_covered(self) -> None:
        """Every ToolCategory should return a valid defaults dict."""
        for category in ToolCategory:
            defaults = _get_category_defaults(category)
            assert len(defaults) == 5


# ---------------------------------------------------------------------------
# _apply_team_size_modifier
# ---------------------------------------------------------------------------


class TestApplyTeamSizeModifier:
    """Tests for the _apply_team_size_modifier helper."""

    def _base_dims(self) -> dict[str, float]:
        return {
            "task_automation_ratio": 5.0,
            "api_openness": 5.0,
            "workflow_complexity": 5.0,
            "data_sensitivity": 5.0,
            "incumbent_inertia": 5.0,
        }

    def test_none_team_size_no_change(self) -> None:
        dims = self._base_dims()
        original_inertia = dims["incumbent_inertia"]
        result = _apply_team_size_modifier(dims, None)
        assert result["incumbent_inertia"] == original_inertia

    def test_large_team_increases_inertia(self) -> None:
        dims = self._base_dims()
        result = _apply_team_size_modifier(dims, 100)
        assert result["incumbent_inertia"] > 5.0

    def test_very_large_team_increases_inertia_more(self) -> None:
        dims_medium = self._base_dims()
        dims_large = self._base_dims()
        _apply_team_size_modifier(dims_medium, 20)
        _apply_team_size_modifier(dims_large, 100)
        assert dims_large["incumbent_inertia"] >= dims_medium["incumbent_inertia"]

    def test_tiny_team_can_decrease_inertia(self) -> None:
        dims = self._base_dims()
        # team_size=1 should trigger the 0-threshold bracket (delta=-0.25)
        result = _apply_team_size_modifier(dims, 1)
        assert result["incumbent_inertia"] <= 5.0

    def test_inertia_clamped_at_10(self) -> None:
        dims = self._base_dims()
        dims["incumbent_inertia"] = 9.5
        result = _apply_team_size_modifier(dims, 200)
        assert result["incumbent_inertia"] <= 10.0

    def test_inertia_clamped_at_0(self) -> None:
        dims = self._base_dims()
        dims["incumbent_inertia"] = 0.1
        result = _apply_team_size_modifier(dims, 1)
        assert result["incumbent_inertia"] >= 0.0

    def test_only_inertia_dimension_modified(self) -> None:
        dims = self._base_dims()
        original = dict(dims)
        _apply_team_size_modifier(dims, 50)
        for key in ("task_automation_ratio", "api_openness", "workflow_complexity", "data_sensitivity"):
            assert dims[key] == original[key]


# ---------------------------------------------------------------------------
# _apply_cost_modifier
# ---------------------------------------------------------------------------


class TestApplyCostModifier:
    """Tests for the _apply_cost_modifier helper."""

    def _base_dims(self) -> dict[str, float]:
        return {
            "task_automation_ratio": 5.0,
            "api_openness": 5.0,
            "workflow_complexity": 5.0,
            "data_sensitivity": 5.0,
            "incumbent_inertia": 5.0,
        }

    def test_none_cost_no_change(self) -> None:
        dims = self._base_dims()
        original = dict(dims)
        _apply_cost_modifier(dims, None)
        assert dims == original

    def test_high_cost_increases_inertia(self) -> None:
        dims = self._base_dims()
        result = _apply_cost_modifier(dims, 5000.0)
        assert result["incumbent_inertia"] > 5.0

    def test_zero_cost_can_decrease_inertia(self) -> None:
        dims = self._base_dims()
        result = _apply_cost_modifier(dims, 0.0)
        assert result["incumbent_inertia"] <= 5.0

    def test_very_high_cost_also_raises_data_sensitivity(self) -> None:
        dims = self._base_dims()
        original_ds = dims["data_sensitivity"]
        result = _apply_cost_modifier(dims, 3000.0)
        assert result["data_sensitivity"] > original_ds

    def test_low_cost_does_not_raise_data_sensitivity(self) -> None:
        dims = self._base_dims()
        original_ds = dims["data_sensitivity"]
        result = _apply_cost_modifier(dims, 50.0)
        assert result["data_sensitivity"] == original_ds

    def test_inertia_clamped(self) -> None:
        dims = self._base_dims()
        dims["incumbent_inertia"] = 9.8
        result = _apply_cost_modifier(dims, 10000.0)
        assert result["incumbent_inertia"] <= 10.0


# ---------------------------------------------------------------------------
# _apply_notes_modifiers
# ---------------------------------------------------------------------------


class TestApplyNotesModifiers:
    """Tests for the _apply_notes_modifiers helper."""

    def _base_dims(self) -> dict[str, float]:
        return {
            "task_automation_ratio": 5.0,
            "api_openness": 5.0,
            "workflow_complexity": 5.0,
            "data_sensitivity": 5.0,
            "incumbent_inertia": 5.0,
        }

    def test_none_notes_no_change(self) -> None:
        dims = self._base_dims()
        original = dict(dims)
        _apply_notes_modifiers(dims, None)
        assert dims == original

    def test_empty_notes_no_change(self) -> None:
        dims = self._base_dims()
        original = dict(dims)
        _apply_notes_modifiers(dims, "   ")
        assert dims == original

    def test_compliance_keyword_raises_data_sensitivity(self) -> None:
        dims = self._base_dims()
        original_ds = dims["data_sensitivity"]
        _apply_notes_modifiers(dims, "This tool handles compliance requirements")
        assert dims["data_sensitivity"] > original_ds

    def test_payroll_keyword_raises_data_sensitivity_significantly(self) -> None:
        dims = self._base_dims()
        original_ds = dims["data_sensitivity"]
        _apply_notes_modifiers(dims, "Used for payroll processing")
        assert dims["data_sensitivity"] > original_ds + 0.5

    def test_api_keyword_raises_api_openness(self) -> None:
        dims = self._base_dims()
        original_api = dims["api_openness"]
        _apply_notes_modifiers(dims, "Has a comprehensive API for integrations")
        assert dims["api_openness"] > original_api

    def test_automation_keyword_raises_task_ratio(self) -> None:
        dims = self._base_dims()
        original_tar = dims["task_automation_ratio"]
        _apply_notes_modifiers(dims, "Used for automation of marketing workflows")
        assert dims["task_automation_ratio"] > original_tar

    def test_custom_keyword_raises_complexity_and_inertia(self) -> None:
        dims = self._base_dims()
        original_wf = dims["workflow_complexity"]
        original_in = dims["incumbent_inertia"]
        _apply_notes_modifiers(dims, "Heavily customized for our use case")
        assert dims["workflow_complexity"] > original_wf
        assert dims["incumbent_inertia"] > original_in

    def test_migration_keyword_lowers_inertia(self) -> None:
        dims = self._base_dims()
        original_in = dims["incumbent_inertia"]
        _apply_notes_modifiers(dims, "Migration to a new system is underway")
        assert dims["incumbent_inertia"] < original_in

    def test_simple_keyword_lowers_workflow_complexity(self) -> None:
        dims = self._base_dims()
        original_wf = dims["workflow_complexity"]
        _apply_notes_modifiers(dims, "Simple lightweight tool")
        assert dims["workflow_complexity"] < original_wf

    def test_case_insensitive_matching(self) -> None:
        dims_lower = self._base_dims()
        dims_upper = self._base_dims()
        _apply_notes_modifiers(dims_lower, "compliance audit")
        _apply_notes_modifiers(dims_upper, "COMPLIANCE AUDIT")
        assert dims_lower["data_sensitivity"] == dims_upper["data_sensitivity"]

    def test_scores_clamped_at_10(self) -> None:
        dims = self._base_dims()
        dims["data_sensitivity"] = 9.9
        # Multiple compliance signals should not push above 10
        notes = "payroll compliance PCI audit regulatory sensitive"
        _apply_notes_modifiers(dims, notes)
        assert dims["data_sensitivity"] <= 10.0

    def test_scores_clamped_at_0(self) -> None:
        dims = self._base_dims()
        dims["incumbent_inertia"] = 0.1
        # Multiple migration/simple signals should not push below 0
        notes = "simple lightweight migration simple basic"
        _apply_notes_modifiers(dims, notes)
        assert dims["incumbent_inertia"] >= 0.0


# ---------------------------------------------------------------------------
# _build_dimensions_from_kb
# ---------------------------------------------------------------------------


class TestBuildDimensionsFromKb:
    """Tests for the _build_dimensions_from_kb function."""

    def test_returns_five_keys(self) -> None:
        entry = lookup("Zapier")
        assert entry is not None
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        dims = _build_dimensions_from_kb(entry, tool)
        expected = {
            "task_automation_ratio", "api_openness", "workflow_complexity",
            "data_sensitivity", "incumbent_inertia"
        }
        assert set(dims.keys()) == expected

    def test_baseline_values_from_entry(self) -> None:
        entry = lookup("Zapier")
        assert entry is not None
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        dims = _build_dimensions_from_kb(entry, tool)
        # Without any metadata modifiers, should start from KB values
        # (may differ slightly if cost/team modifiers are applied)
        # At minimum, task_automation_ratio should stay close to KB value
        assert dims["task_automation_ratio"] >= entry.task_automation_ratio - 1.0

    def test_team_size_modifier_applied(self) -> None:
        entry = lookup("Zapier")
        assert entry is not None
        tool_no_team = make_tool("Zapier", ToolCategory.AUTOMATION)
        tool_large_team = make_tool("Zapier", ToolCategory.AUTOMATION, team_size=100)
        dims_no = _build_dimensions_from_kb(entry, tool_no_team)
        dims_large = _build_dimensions_from_kb(entry, tool_large_team)
        assert dims_large["incumbent_inertia"] > dims_no["incumbent_inertia"]

    def test_cost_modifier_applied(self) -> None:
        entry = lookup("Zapier")
        assert entry is not None
        tool_cheap = make_tool("Zapier", ToolCategory.AUTOMATION, monthly_cost_usd=10.0)
        tool_expensive = make_tool("Zapier", ToolCategory.AUTOMATION, monthly_cost_usd=5000.0)
        dims_cheap = _build_dimensions_from_kb(entry, tool_cheap)
        dims_exp = _build_dimensions_from_kb(entry, tool_expensive)
        assert dims_exp["incumbent_inertia"] > dims_cheap["incumbent_inertia"]

    def test_all_scores_in_range(self) -> None:
        entry = lookup("Salesforce")
        assert entry is not None
        tool = make_tool("Salesforce", ToolCategory.CRM, monthly_cost_usd=3200.0, team_size=35)
        dims = _build_dimensions_from_kb(entry, tool)
        for key, val in dims.items():
            assert 0.0 <= val <= 10.0, f"Dimension '{key}' out of range: {val}"


# ---------------------------------------------------------------------------
# _build_dimensions_from_defaults
# ---------------------------------------------------------------------------


class TestBuildDimensionsFromDefaults:
    """Tests for the _build_dimensions_from_defaults function."""

    def test_returns_five_keys(self) -> None:
        tool = make_tool("UnknownTool", ToolCategory.AUTOMATION)
        dims = _build_dimensions_from_defaults(tool)
        expected = {
            "task_automation_ratio", "api_openness", "workflow_complexity",
            "data_sensitivity", "incumbent_inertia"
        }
        assert set(dims.keys()) == expected

    def test_uses_category_defaults(self) -> None:
        tool = make_tool("UnknownTool", ToolCategory.AUTOMATION)
        dims = _build_dimensions_from_defaults(tool)
        category_defaults = _get_category_defaults(ToolCategory.AUTOMATION)
        # Without modifiers, should match category defaults
        # (small differences possible from rounding)
        assert abs(dims["task_automation_ratio"] - category_defaults["task_automation_ratio"]) <= 2.0

    def test_all_scores_in_range(self) -> None:
        for category in ToolCategory:
            tool = make_tool("UnknownTool", category)
            dims = _build_dimensions_from_defaults(tool)
            for key, val in dims.items():
                assert 0.0 <= val <= 10.0, (
                    f"Category {category} dimension '{key}' out of range: {val}"
                )

    def test_metadata_modifiers_applied(self) -> None:
        tool_plain = make_tool("UnknownTool", ToolCategory.HR)
        tool_with_meta = make_tool(
            "UnknownTool", ToolCategory.HR,
            team_size=150,
            monthly_cost_usd=5000.0,
            notes="Handles payroll and compliance",
        )
        dims_plain = _build_dimensions_from_defaults(tool_plain)
        dims_meta = _build_dimensions_from_defaults(tool_with_meta)
        # With large team, high cost, and compliance notes → higher inertia/sensitivity
        assert dims_meta["incumbent_inertia"] >= dims_plain["incumbent_inertia"]
        assert dims_meta["data_sensitivity"] >= dims_plain["data_sensitivity"]


# ---------------------------------------------------------------------------
# _get_default_alternatives
# ---------------------------------------------------------------------------


class TestGetDefaultAlternatives:
    """Tests for _get_default_alternatives."""

    def test_returns_list(self) -> None:
        alts = _get_default_alternatives(ToolCategory.AUTOMATION)
        assert isinstance(alts, list)
        assert len(alts) > 0

    def test_all_categories_have_alternatives(self) -> None:
        for category in ToolCategory:
            alts = _get_default_alternatives(category)
            assert len(alts) > 0, f"Category {category} has no default alternatives"

    def test_returns_copy(self) -> None:
        alts1 = _get_default_alternatives(ToolCategory.AUTOMATION)
        alts1.clear()
        alts2 = _get_default_alternatives(ToolCategory.AUTOMATION)
        assert len(alts2) > 0


# ---------------------------------------------------------------------------
# score_tool
# ---------------------------------------------------------------------------


class TestScoreTool:
    """Tests for the score_tool() function."""

    def test_returns_analysis_result(self) -> None:
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert isinstance(result, AnalysisResult)

    def test_known_tool_uses_kb_source(self) -> None:
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.source == "knowledge_base"

    def test_unknown_tool_uses_default_source(self) -> None:
        tool = make_tool("CompletelyUnknownTool12345", ToolCategory.OTHER)
        result = score_tool(tool)
        assert result.source == "default"

    def test_known_tool_has_alternatives(self) -> None:
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert len(result.score.alternatives) > 0

    def test_unknown_tool_has_alternatives(self) -> None:
        tool = make_tool("UnknownTool", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert len(result.score.alternatives) > 0

    def test_known_tool_has_rationale(self) -> None:
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.score.rationale is not None
        assert len(result.score.rationale) > 0

    def test_unknown_tool_has_rationale(self) -> None:
        tool = make_tool("UnknownTool", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.score.rationale is not None
        assert len(result.score.rationale) > 0

    def test_displacement_score_in_range(self) -> None:
        for category in ToolCategory:
            tool = make_tool("TestTool", category)
            result = score_tool(tool)
            assert 0.0 <= result.displacement_score <= 100.0

    def test_enriched_by_llm_false(self) -> None:
        """Rule-based scorer should never set enriched_by_llm=True."""
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.score.enriched_by_llm is False

    def test_disable_kb_uses_default(self) -> None:
        """With use_knowledge_base=False, even known tools use defaults."""
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool, use_knowledge_base=False)
        assert result.source == "default"

    def test_rank_not_set(self) -> None:
        """score_tool does not set rank; that's done in build_report."""
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.rank is None

    def test_tool_preserved_on_result(self) -> None:
        tool = make_tool("Zapier", ToolCategory.AUTOMATION, monthly_cost_usd=599.0)
        result = score_tool(tool)
        assert result.tool.name == "Zapier"
        assert result.tool.monthly_cost_usd == 599.0

    def test_zapier_is_high_risk(self) -> None:
        """Zapier should score as Critical or High risk."""
        tool = make_tool("Zapier", ToolCategory.AUTOMATION)
        result = score_tool(tool)
        assert result.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)

    def test_workday_is_low_medium_risk(self) -> None:
        """Workday's high data sensitivity and inertia should keep risk lower."""
        tool = make_tool("Workday", ToolCategory.HR)
        result = score_tool(tool)
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_metadata_modifiers_affect_score(self) -> None:
        """Adding large team and high cost should affect the result score."""
        tool_plain = make_tool("UnknownCRM", ToolCategory.CRM)
        tool_big = make_tool(
            "UnknownCRM", ToolCategory.CRM,
            team_size=200,
            monthly_cost_usd=10000.0,
            notes="Heavily customized enterprise CRM with compliance requirements",
        )
        result_plain = score_tool(tool_plain)
        result_big = score_tool(tool_big)
        # Bigger/more complex deployment should generally score differently
        # (higher inertia/sensitivity should lower the displacement score)
        assert result_big.score.incumbent_inertia >= result_plain.score.incumbent_inertia

    def test_case_insensitive_kb_lookup(self) -> None:
        """KB lookup should be case-insensitive."""
        tool_lower = make_tool("zapier", ToolCategory.AUTOMATION)
        tool_upper = make_tool("ZAPIER", ToolCategory.AUTOMATION)
        result_lower = score_tool(tool_lower)
        result_upper = score_tool(tool_upper)
        # Both should hit the KB
        assert result_lower.source == "knowledge_base"
        assert result_upper.source == "knowledge_base"

    def test_all_known_tools_score_without_error(self) -> None:
        """All tools in the KB should score successfully."""
        from saas_risk_scan.knowledge_base import all_entries
        for entry in all_entries():
            tool = make_tool(entry.name, entry.category)
            result = score_tool(tool)
            assert 0.0 <= result.displacement_score <= 100.0


# ---------------------------------------------------------------------------
# score_stack
# ---------------------------------------------------------------------------


class TestScoreStack:
    """Tests for the score_stack() function."""

    def test_returns_one_result_per_tool(self) -> None:
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("Notion", ToolCategory.KNOWLEDGE_MANAGEMENT),
            make_tool("Salesforce", ToolCategory.CRM),
        ])
        results = score_stack(stack)
        assert len(results) == 3

    def test_all_results_are_analysis_result(self) -> None:
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("UnknownTool", ToolCategory.OTHER),
        ])
        results = score_stack(stack)
        for r in results:
            assert isinstance(r, AnalysisResult)

    def test_preserves_tool_order(self) -> None:
        tools = [
            make_tool("Alpha", ToolCategory.OTHER),
            make_tool("Beta", ToolCategory.OTHER),
            make_tool("Gamma", ToolCategory.OTHER),
        ]
        stack = SaasStack(tools=tools)
        results = score_stack(stack)
        assert results[0].tool.name == "Alpha"
        assert results[1].tool.name == "Beta"
        assert results[2].tool.name == "Gamma"

    def test_single_tool_stack(self) -> None:
        stack = SaasStack(tools=[make_tool("Zapier", ToolCategory.AUTOMATION)])
        results = score_stack(stack)
        assert len(results) == 1

    def test_mixed_known_and_unknown_tools(self) -> None:
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),  # known
            make_tool("MagicUnknownSaaS", ToolCategory.CRM),  # unknown
        ])
        results = score_stack(stack)
        sources = {r.source for r in results}
        assert "knowledge_base" in sources
        assert "default" in sources

    def test_all_displacement_scores_valid(self) -> None:
        stack = SaasStack(tools=[
            make_tool(name, cat)
            for name, cat in [
                ("Zapier", ToolCategory.AUTOMATION),
                ("Salesforce", ToolCategory.CRM),
                ("Notion", ToolCategory.KNOWLEDGE_MANAGEMENT),
                ("Workday", ToolCategory.HR),
                ("Stripe", ToolCategory.FINANCE),
            ]
        ])
        results = score_stack(stack)
        for r in results:
            assert 0.0 <= r.displacement_score <= 100.0


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Tests for the build_report() function."""

    def _make_stack(self) -> SaasStack:
        return SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION, monthly_cost_usd=599.0),
            make_tool("Salesforce", ToolCategory.CRM, monthly_cost_usd=3200.0),
            make_tool("Notion", ToolCategory.KNOWLEDGE_MANAGEMENT, monthly_cost_usd=320.0),
        ])

    def test_returns_scan_report(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        assert isinstance(report, ScanReport)

    def test_report_has_correct_tool_count(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        assert report.total_tools == 3

    def test_results_are_ranked(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        for i, result in enumerate(report.results, start=1):
            assert result.rank == i

    def test_results_sorted_descending(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        scores = [r.displacement_score for r in report.results]
        assert scores == sorted(scores, reverse=True)

    def test_summary_stats_populated(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        assert "avg_displacement_score" in report.summary_stats
        assert "risk_level_counts" in report.summary_stats
        assert "total_monthly_cost_usd" in report.summary_stats

    def test_generated_at_preserved(self) -> None:
        stack = self._make_stack()
        ts = "2024-06-15T12:00:00"
        report = build_report(stack, generated_at=ts)
        assert report.generated_at == ts

    def test_enrichment_enabled_flag(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00", enrichment_enabled=True)
        assert report.enrichment_enabled is True

    def test_enrichment_disabled_by_default(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        assert report.enrichment_enabled is False

    def test_total_monthly_cost_in_summary(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        expected_cost = 599.0 + 3200.0 + 320.0
        assert report.summary_stats["total_monthly_cost_usd"] == expected_cost

    def test_pre_scored_results_merged(self) -> None:
        """Pre-scored results are used; remaining tools are scored by rule engine."""
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("Notion", ToolCategory.KNOWLEDGE_MANAGEMENT),
            make_tool("Salesforce", ToolCategory.CRM),
        ])
        # Pre-score only Zapier
        zapier_result = score_tool(make_tool("Zapier", ToolCategory.AUTOMATION))
        report = build_report(
            stack,
            generated_at="2024-01-01T00:00:00",
            pre_scored_results=[zapier_result],
        )
        assert report.total_tools == 3
        tool_names = {r.tool.name for r in report.results}
        assert "Zapier" in tool_names
        assert "Notion" in tool_names
        assert "Salesforce" in tool_names

    def test_pre_scored_no_duplicate_tools(self) -> None:
        """Tools already in pre_scored_results are not double-scored."""
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
        ])
        zapier_result = score_tool(make_tool("Zapier", ToolCategory.AUTOMATION))
        report = build_report(
            stack,
            generated_at="2024-01-01T00:00:00",
            pre_scored_results=[zapier_result],
        )
        # Should have exactly 1 result (not 2)
        assert report.total_tools == 1

    def test_to_dict_contains_results(self) -> None:
        stack = self._make_stack()
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        d = report.to_dict()
        assert len(d["results"]) == 3

    def test_single_tool_stack(self) -> None:
        stack = SaasStack(tools=[make_tool("Zapier", ToolCategory.AUTOMATION)])
        report = build_report(stack, generated_at="2024-01-01T00:00:00")
        assert report.total_tools == 1
        assert report.results[0].rank == 1


# ---------------------------------------------------------------------------
# get_dimension_weights
# ---------------------------------------------------------------------------


class TestGetDimensionWeights:
    """Tests for get_dimension_weights()."""

    def test_returns_dict_with_five_keys(self) -> None:
        weights = get_dimension_weights()
        expected = {
            "task_automation_ratio",
            "api_openness",
            "workflow_complexity",
            "data_sensitivity",
            "incumbent_inertia",
        }
        assert set(weights.keys()) == expected

    def test_weights_sum_to_100(self) -> None:
        weights = get_dimension_weights()
        assert sum(weights.values()) == 100.0

    def test_task_automation_ratio_highest_weight(self) -> None:
        weights = get_dimension_weights()
        assert weights["task_automation_ratio"] == max(weights.values())

    def test_all_weights_positive(self) -> None:
        weights = get_dimension_weights()
        for name, w in weights.items():
            assert w > 0.0, f"Weight for '{name}' should be positive"


# ---------------------------------------------------------------------------
# score_dimensions_only
# ---------------------------------------------------------------------------


class TestScoreDimensionsOnly:
    """Tests for the score_dimensions_only() utility function."""

    def test_returns_dict_with_required_keys(self) -> None:
        result = score_dimensions_only(5.0, 5.0, 5.0, 5.0, 5.0)
        required = {
            "task_automation_ratio", "api_openness", "workflow_complexity",
            "data_sensitivity", "incumbent_inertia",
            "displacement_score", "risk_level", "timeline", "timeline_display"
        }
        assert set(result.keys()) == required

    def test_midpoint_scores_give_50(self) -> None:
        result = score_dimensions_only(5.0, 5.0, 5.0, 5.0, 5.0)
        assert result["displacement_score"] == 50.0

    def test_maximum_scores_give_100(self) -> None:
        result = score_dimensions_only(10.0, 10.0, 0.0, 0.0, 0.0)
        assert result["displacement_score"] == 100.0

    def test_minimum_scores_give_0(self) -> None:
        result = score_dimensions_only(0.0, 0.0, 10.0, 10.0, 10.0)
        assert result["displacement_score"] == 0.0

    def test_risk_level_is_string(self) -> None:
        result = score_dimensions_only(5.0, 5.0, 5.0, 5.0, 5.0)
        assert isinstance(result["risk_level"], str)

    def test_timeline_is_string(self) -> None:
        result = score_dimensions_only(5.0, 5.0, 5.0, 5.0, 5.0)
        assert isinstance(result["timeline"], str)

    def test_timeline_display_is_string(self) -> None:
        result = score_dimensions_only(5.0, 5.0, 5.0, 5.0, 5.0)
        assert isinstance(result["timeline_display"], str)
        assert len(result["timeline_display"]) > 0

    def test_critical_risk_at_high_scores(self) -> None:
        result = score_dimensions_only(9.0, 9.0, 1.0, 1.0, 1.0)
        assert result["risk_level"] == "critical"
        assert result["timeline"] == "near"

    def test_low_risk_at_low_scores(self) -> None:
        result = score_dimensions_only(1.0, 1.0, 9.0, 9.0, 9.0)
        assert result["risk_level"] == "low"
        assert result["timeline"] == "unlikely"

    def test_input_dimensions_preserved_in_output(self) -> None:
        result = score_dimensions_only(7.5, 6.0, 3.0, 4.0, 2.0)
        assert result["task_automation_ratio"] == 7.5
        assert result["api_openness"] == 6.0
        assert result["workflow_complexity"] == 3.0
        assert result["data_sensitivity"] == 4.0
        assert result["incumbent_inertia"] == 2.0


# ---------------------------------------------------------------------------
# Integration: end-to-end stack scoring
# ---------------------------------------------------------------------------


class TestIntegrationStackScoring:
    """End-to-end integration tests for the full scoring pipeline."""

    def test_sample_stack_scores_without_error(self) -> None:
        """Scoring the example SaaS stack should complete without errors."""
        from pathlib import Path
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        import datetime
        stack = load_file(sample)
        report = build_report(
            stack,
            generated_at=datetime.datetime.now().isoformat(),
        )
        assert report.total_tools == stack.tool_count()
        assert len(report.results) == stack.tool_count()
        for r in report.results:
            assert r.rank is not None
            assert 0.0 <= r.displacement_score <= 100.0

    def test_report_rank_is_unique(self) -> None:
        """Each result in a report should have a unique rank."""
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("Salesforce", ToolCategory.CRM),
            make_tool("Workday", ToolCategory.HR),
            make_tool("GitHub", ToolCategory.DEVTOOLS),
        ])
        import datetime
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        ranks = [r.rank for r in report.results]
        assert len(ranks) == len(set(ranks)), "Ranks should be unique"

    def test_avg_score_in_summary(self) -> None:
        """Average score in summary should match manual calculation."""
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("Workday", ToolCategory.HR),
        ])
        import datetime
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        scores = [r.displacement_score for r in report.results]
        expected_avg = round(sum(scores) / len(scores), 1)
        assert report.summary_stats["avg_displacement_score"] == expected_avg

    def test_risk_level_counts_correct(self) -> None:
        """Risk level counts in summary should match actual results."""
        stack = SaasStack(tools=[
            make_tool("Zapier", ToolCategory.AUTOMATION),
            make_tool("Salesforce", ToolCategory.CRM),
            make_tool("Workday", ToolCategory.HR),
        ])
        import datetime
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        actual_counts: dict[str, int] = {}
        for r in report.results:
            level = r.risk_level.value
            actual_counts[level] = actual_counts.get(level, 0) + 1
        assert report.summary_stats["risk_level_counts"] == actual_counts
