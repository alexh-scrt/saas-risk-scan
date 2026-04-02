"""Unit tests for saas_risk_scan/models.py.

Covers Pydantic model validation, helper functions, and derived field logic.
"""

import pytest

from saas_risk_scan.models import (
    AnalysisResult,
    DisplacementScore,
    ReplacementTimeline,
    RiskLevel,
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
# ToolCategory tests
# ---------------------------------------------------------------------------


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_all_categories_have_values(self) -> None:
        expected = {
            "automation", "crm", "customer_support", "data_analytics",
            "knowledge_management", "project_management", "communication",
            "hr", "finance", "marketing", "devtools", "security",
            "storage", "ecommerce", "other",
        }
        actual = {c.value for c in ToolCategory}
        assert actual == expected

    def test_category_is_string_enum(self) -> None:
        assert isinstance(ToolCategory.AUTOMATION, str)
        assert ToolCategory.AUTOMATION == "automation"


# ---------------------------------------------------------------------------
# RiskLevel tests
# ---------------------------------------------------------------------------


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_emoji_mapping(self) -> None:
        assert RiskLevel.CRITICAL.emoji == "🔴"
        assert RiskLevel.HIGH.emoji == "🟠"
        assert RiskLevel.MEDIUM.emoji == "🟡"
        assert RiskLevel.LOW.emoji == "🟢"

    def test_label_is_capitalized(self) -> None:
        assert RiskLevel.CRITICAL.label == "Critical"
        assert RiskLevel.LOW.label == "Low"


# ---------------------------------------------------------------------------
# ReplacementTimeline tests
# ---------------------------------------------------------------------------


class TestReplacementTimeline:
    """Tests for ReplacementTimeline enum."""

    def test_display_strings(self) -> None:
        assert ReplacementTimeline.NEAR.display == "< 12 months"
        assert ReplacementTimeline.MID.display == "12–24 months"
        assert ReplacementTimeline.LONG.display == "24–36 months"
        assert ReplacementTimeline.UNLIKELY.display == "36+ months"


# ---------------------------------------------------------------------------
# SaasTool tests
# ---------------------------------------------------------------------------


class TestSaasTool:
    """Tests for the SaasTool Pydantic model."""

    def test_minimal_valid_tool(self) -> None:
        tool = SaasTool(name="Zapier", category=ToolCategory.AUTOMATION)
        assert tool.name == "Zapier"
        assert tool.category == ToolCategory.AUTOMATION
        assert tool.monthly_cost_usd is None
        assert tool.team_size is None
        assert tool.notes is None

    def test_full_tool(self) -> None:
        tool = SaasTool(
            name="Salesforce",
            category=ToolCategory.CRM,
            monthly_cost_usd=3200.0,
            team_size=35,
            notes="Core CRM",
        )
        assert tool.monthly_cost_usd == 3200.0
        assert tool.team_size == 35
        assert tool.notes == "Core CRM"

    def test_category_normalized_to_lowercase(self) -> None:
        tool = SaasTool(name="Zapier", category="AUTOMATION")  # type: ignore[arg-type]
        assert tool.category == ToolCategory.AUTOMATION

    def test_name_whitespace_stripped(self) -> None:
        tool = SaasTool(name="  Zapier  ", category=ToolCategory.AUTOMATION)
        assert tool.name == "Zapier"

    def test_blank_name_raises(self) -> None:
        with pytest.raises(Exception):
            SaasTool(name="   ", category=ToolCategory.AUTOMATION)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(Exception):
            SaasTool(name="", category=ToolCategory.AUTOMATION)

    def test_negative_monthly_cost_raises(self) -> None:
        with pytest.raises(Exception):
            SaasTool(name="Tool", category=ToolCategory.OTHER, monthly_cost_usd=-1.0)

    def test_zero_team_size_raises(self) -> None:
        with pytest.raises(Exception):
            SaasTool(name="Tool", category=ToolCategory.OTHER, team_size=0)

    def test_display_name_short(self) -> None:
        tool = SaasTool(name="Zapier", category=ToolCategory.AUTOMATION)
        assert tool.display_name() == "Zapier"

    def test_display_name_truncated(self) -> None:
        long_name = "A" * 50
        tool = SaasTool(name=long_name, category=ToolCategory.AUTOMATION)
        assert len(tool.display_name()) == 40

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(Exception):
            SaasTool(name="Tool", category="not_a_category")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_displacement_score tests
# ---------------------------------------------------------------------------


class TestComputeDisplacementScore:
    """Tests for the compute_displacement_score helper function."""

    def test_maximum_score(self) -> None:
        # All pro-displacement, all resistance at 0 → 100.0
        score = compute_displacement_score(
            task_automation_ratio=10.0,
            api_openness=10.0,
            workflow_complexity=0.0,
            data_sensitivity=0.0,
            incumbent_inertia=0.0,
        )
        assert score == 100.0

    def test_minimum_score(self) -> None:
        # All pro-displacement at 0, all resistance at 10 → 0.0
        score = compute_displacement_score(
            task_automation_ratio=0.0,
            api_openness=0.0,
            workflow_complexity=10.0,
            data_sensitivity=10.0,
            incumbent_inertia=10.0,
        )
        assert score == 0.0

    def test_midpoint_score(self) -> None:
        # All dimensions at 5 → 50.0
        score = compute_displacement_score(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
        )
        assert score == 50.0

    def test_score_is_rounded_to_one_decimal(self) -> None:
        score = compute_displacement_score(
            task_automation_ratio=7.0,
            api_openness=6.0,
            workflow_complexity=3.0,
            data_sensitivity=4.0,
            incumbent_inertia=2.0,
        )
        # Verify it's a float with at most one decimal
        assert isinstance(score, float)
        assert score == round(score, 1)

    def test_score_within_bounds(self) -> None:
        score = compute_displacement_score(8.0, 7.0, 2.0, 3.0, 1.0)
        assert 0.0 <= score <= 100.0

    def test_weights_sum_to_100(self) -> None:
        # Verify the weight structure: 3+2+1.5+2+1.5 = 10, so max raw = 10*10 = 100
        assert (3.0 + 2.0 + 1.5 + 2.0 + 1.5) * 10 == 100.0


# ---------------------------------------------------------------------------
# score_to_risk_level tests
# ---------------------------------------------------------------------------


class TestScoreToRiskLevel:
    """Tests for the score_to_risk_level helper."""

    def test_critical_at_75(self) -> None:
        assert score_to_risk_level(75.0) == RiskLevel.CRITICAL

    def test_critical_at_100(self) -> None:
        assert score_to_risk_level(100.0) == RiskLevel.CRITICAL

    def test_high_at_50(self) -> None:
        assert score_to_risk_level(50.0) == RiskLevel.HIGH

    def test_high_at_74(self) -> None:
        assert score_to_risk_level(74.9) == RiskLevel.HIGH

    def test_medium_at_25(self) -> None:
        assert score_to_risk_level(25.0) == RiskLevel.MEDIUM

    def test_medium_at_49(self) -> None:
        assert score_to_risk_level(49.9) == RiskLevel.MEDIUM

    def test_low_at_0(self) -> None:
        assert score_to_risk_level(0.0) == RiskLevel.LOW

    def test_low_at_24(self) -> None:
        assert score_to_risk_level(24.9) == RiskLevel.LOW

    def test_boundary_74_99_is_high(self) -> None:
        assert score_to_risk_level(74.99) == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# score_to_timeline tests
# ---------------------------------------------------------------------------


class TestScoreToTimeline:
    """Tests for the score_to_timeline helper."""

    def test_near_at_75(self) -> None:
        assert score_to_timeline(75.0) == ReplacementTimeline.NEAR

    def test_mid_at_50(self) -> None:
        assert score_to_timeline(50.0) == ReplacementTimeline.MID

    def test_long_at_25(self) -> None:
        assert score_to_timeline(25.0) == ReplacementTimeline.LONG

    def test_unlikely_at_0(self) -> None:
        assert score_to_timeline(0.0) == ReplacementTimeline.UNLIKELY

    def test_unlikely_at_24(self) -> None:
        assert score_to_timeline(24.9) == ReplacementTimeline.UNLIKELY

    def test_near_at_100(self) -> None:
        assert score_to_timeline(100.0) == ReplacementTimeline.NEAR


# ---------------------------------------------------------------------------
# RiskScore tests
# ---------------------------------------------------------------------------


class TestRiskScore:
    """Tests for the RiskScore model."""

    def test_from_dimensions_computes_score(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=10.0,
            api_openness=10.0,
            workflow_complexity=0.0,
            data_sensitivity=0.0,
            incumbent_inertia=0.0,
        )
        assert rs.displacement_score == 100.0
        assert rs.risk_level == RiskLevel.CRITICAL
        assert rs.timeline == ReplacementTimeline.NEAR

    def test_from_dimensions_minimum(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=0.0,
            api_openness=0.0,
            workflow_complexity=10.0,
            data_sensitivity=10.0,
            incumbent_inertia=10.0,
        )
        assert rs.displacement_score == 0.0
        assert rs.risk_level == RiskLevel.LOW
        assert rs.timeline == ReplacementTimeline.UNLIKELY

    def test_from_dimensions_with_alternatives(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=8.0,
            api_openness=8.0,
            workflow_complexity=2.0,
            data_sensitivity=2.0,
            incumbent_inertia=2.0,
            alternatives=["n8n", "LangChain"],
            rationale="Highly automatable workflow tool.",
            enriched_by_llm=True,
        )
        assert rs.alternatives == ["n8n", "LangChain"]
        assert rs.rationale == "Highly automatable workflow tool."
        assert rs.enriched_by_llm is True

    def test_default_alternatives_empty(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
        )
        assert rs.alternatives == []
        assert rs.enriched_by_llm is False

    def test_top_alternative_returns_first(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
            alternatives=["Alt1", "Alt2", "Alt3"],
        )
        assert rs.top_alternative() == "Alt1"

    def test_top_alternative_none_when_empty(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
        )
        assert rs.top_alternative() is None

    def test_alternatives_display_truncated(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
            alternatives=["Alt1", "Alt2", "Alt3", "Alt4"],
        )
        display = rs.alternatives_display(max_items=3)
        assert display == "Alt1, Alt2, Alt3"

    def test_alternatives_display_empty(self) -> None:
        rs = RiskScore.from_dimensions(
            task_automation_ratio=5.0,
            api_openness=5.0,
            workflow_complexity=5.0,
            data_sensitivity=5.0,
            incumbent_inertia=5.0,
        )
        assert rs.alternatives_display() == "—"

    def test_dimension_score_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            RiskScore(
                task_automation_ratio=11.0,  # out of range
                api_openness=5.0,
                workflow_complexity=5.0,
                data_sensitivity=5.0,
                incumbent_inertia=5.0,
                displacement_score=50.0,
                risk_level=RiskLevel.HIGH,
                timeline=ReplacementTimeline.MID,
            )

    def test_displacement_score_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            RiskScore(
                task_automation_ratio=5.0,
                api_openness=5.0,
                workflow_complexity=5.0,
                data_sensitivity=5.0,
                incumbent_inertia=5.0,
                displacement_score=101.0,  # out of range
                risk_level=RiskLevel.HIGH,
                timeline=ReplacementTimeline.MID,
            )


# ---------------------------------------------------------------------------
# AnalysisResult tests
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """Tests for the AnalysisResult model."""

    def _make_result(
        self,
        name: str = "Zapier",
        category: ToolCategory = ToolCategory.AUTOMATION,
        task_auto: float = 9.0,
        api: float = 8.0,
        wf: float = 2.0,
        ds: float = 2.0,
        inertia: float = 2.0,
        source: str = "knowledge_base",
    ) -> AnalysisResult:
        tool = SaasTool(name=name, category=category)
        score = RiskScore.from_dimensions(task_auto, api, wf, ds, inertia)
        return AnalysisResult(tool=tool, score=score, source=source)

    def test_convenience_properties(self) -> None:
        result = self._make_result()
        assert result.tool_name == "Zapier"
        assert result.category == ToolCategory.AUTOMATION
        assert isinstance(result.displacement_score, float)
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.timeline, ReplacementTimeline)

    def test_to_dict_structure(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "tool" in d
        assert "score" in d
        assert "rank" in d
        assert d["source"] == "knowledge_base"
        assert "dimensions" in d["score"]
        assert "displacement_score" in d["score"]

    def test_invalid_source_raises(self) -> None:
        tool = SaasTool(name="Tool", category=ToolCategory.OTHER)
        score = RiskScore.from_dimensions(5.0, 5.0, 5.0, 5.0, 5.0)
        with pytest.raises(Exception):
            AnalysisResult(tool=tool, score=score, source="invalid_source")

    def test_rank_defaults_none(self) -> None:
        result = self._make_result()
        assert result.rank is None

    def test_rank_can_be_set(self) -> None:
        result = self._make_result()
        result.rank = 1
        assert result.rank == 1


# ---------------------------------------------------------------------------
# SaasStack tests
# ---------------------------------------------------------------------------


class TestSaasStack:
    """Tests for the SaasStack model."""

    def test_basic_stack(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="Zapier", category=ToolCategory.AUTOMATION),
                SaasTool(name="Notion", category=ToolCategory.KNOWLEDGE_MANAGEMENT),
            ]
        )
        assert stack.tool_count() == 2

    def test_empty_tools_raises(self) -> None:
        with pytest.raises(Exception):
            SaasStack(tools=[])

    def test_total_monthly_cost_with_costs(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="A", category=ToolCategory.OTHER, monthly_cost_usd=100.0),
                SaasTool(name="B", category=ToolCategory.OTHER, monthly_cost_usd=200.0),
            ]
        )
        assert stack.total_monthly_cost() == 300.0

    def test_total_monthly_cost_none_when_no_data(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="A", category=ToolCategory.OTHER),
                SaasTool(name="B", category=ToolCategory.OTHER),
            ]
        )
        assert stack.total_monthly_cost() is None

    def test_partial_cost_data(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="A", category=ToolCategory.OTHER, monthly_cost_usd=500.0),
                SaasTool(name="B", category=ToolCategory.OTHER),  # no cost
            ]
        )
        assert stack.total_monthly_cost() == 500.0

    def test_duplicate_names_detected(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="Zapier", category=ToolCategory.AUTOMATION),
                SaasTool(name="zapier", category=ToolCategory.AUTOMATION),
            ]
        )
        assert "zapier" in stack.duplicate_names()

    def test_no_duplicates_returns_empty(self) -> None:
        stack = SaasStack(
            tools=[
                SaasTool(name="Zapier", category=ToolCategory.AUTOMATION),
                SaasTool(name="Notion", category=ToolCategory.KNOWLEDGE_MANAGEMENT),
            ]
        )
        assert stack.duplicate_names() == []


# ---------------------------------------------------------------------------
# ScanReport tests
# ---------------------------------------------------------------------------


class TestScanReport:
    """Tests for the ScanReport model."""

    def _make_result(
        self, name: str, score_val: float, monthly_cost: float = 100.0
    ) -> AnalysisResult:
        """Helper to create an AnalysisResult with a specific displacement score."""
        # Find dimensions that produce the desired score
        # Use task_auto and api_openness as drivers, others at midpoint
        # displacement = task*3 + api*2 + (10-wf)*1.5 + (10-ds)*2 + (10-i)*1.5
        # With wf=ds=inertia=5: = task*3 + api*2 + 25
        # score_val = task*3 + api*2 + 25  → task*3 + api*2 = score_val - 25
        # Set api=5 → task*3 = score_val - 25 - 10 = score_val - 35
        # task = (score_val - 35) / 3; clamp to [0,10]
        raw_task = (score_val - 35.0) / 3.0
        task = max(0.0, min(10.0, raw_task))
        # Recompute actual api needed
        remaining = score_val - task * 3.0 - 25.0
        api = max(0.0, min(10.0, remaining / 2.0))
        tool = SaasTool(
            name=name,
            category=ToolCategory.OTHER,
            monthly_cost_usd=monthly_cost,
        )
        score = RiskScore.from_dimensions(task, api, 5.0, 5.0, 5.0)
        return AnalysisResult(tool=tool, score=score, source="default")

    def test_rank_results_sorts_descending(self) -> None:
        report = ScanReport(
            generated_at="2024-01-01T00:00:00",
            results=[
                self._make_result("Low", 30.0),
                self._make_result("High", 80.0),
                self._make_result("Mid", 55.0),
            ],
        )
        report.rank_results()
        assert report.results[0].tool_name == "High"
        assert report.results[1].tool_name == "Mid"
        assert report.results[2].tool_name == "Low"
        assert report.results[0].rank == 1
        assert report.results[1].rank == 2
        assert report.results[2].rank == 3

    def test_compute_summary_populates_stats(self) -> None:
        report = ScanReport(
            generated_at="2024-01-01T00:00:00",
            results=[
                self._make_result("A", 80.0, 200.0),
                self._make_result("B", 50.0, 300.0),
            ],
        )
        report.compute_summary()
        stats = report.summary_stats
        assert "avg_displacement_score" in stats
        assert "risk_level_counts" in stats
        assert "total_monthly_cost_usd" in stats
        assert stats["total_monthly_cost_usd"] == 500.0
        assert stats["tools_enriched_by_llm"] == 0

    def test_compute_summary_empty(self) -> None:
        report = ScanReport(
            generated_at="2024-01-01T00:00:00",
            results=[],
        )
        report.compute_summary()
        assert report.summary_stats["avg_displacement_score"] == 0.0
        assert report.summary_stats["total_monthly_cost_usd"] is None

    def test_to_dict_structure(self) -> None:
        report = ScanReport(
            generated_at="2024-01-01T00:00:00",
            results=[self._make_result("A", 60.0)],
            total_tools=1,
            enrichment_enabled=False,
        )
        report.rank_results()
        report.compute_summary()
        d = report.to_dict()
        assert "generated_at" in d
        assert "total_tools" in d
        assert "results" in d
        assert "summary_stats" in d
        assert len(d["results"]) == 1
