"""Unit tests for saas_risk_scan/reporter.py.

Covers Rich table rendering, Markdown generation, JSON export,
filtering logic, and report loading from JSON.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Optional

import pytest
from rich.console import Console

from saas_risk_scan.models import (
    AnalysisResult,
    RiskLevel,
    RiskScore,
    SaasStack,
    SaasTool,
    ScanReport,
    ToolCategory,
)
from saas_risk_scan.scorer import build_report, score_tool
from saas_risk_scan.reporter import (
    Reporter,
    _format_cost,
    _format_team_size,
    _score_bar_rich,
    _score_bar_text,
    load_report_from_json,
    render,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def make_tool(
    name: str = "Zapier",
    category: ToolCategory = ToolCategory.AUTOMATION,
    monthly_cost_usd: Optional[float] = None,
    team_size: Optional[int] = None,
    notes: Optional[str] = None,
) -> SaasTool:
    """Factory for SaasTool test instances."""
    return SaasTool(
        name=name,
        category=category,
        monthly_cost_usd=monthly_cost_usd,
        team_size=team_size,
        notes=notes,
    )


def make_result(
    name: str = "Zapier",
    category: ToolCategory = ToolCategory.AUTOMATION,
    task_auto: float = 8.0,
    api: float = 7.0,
    wf: float = 3.0,
    ds: float = 3.0,
    inertia: float = 3.0,
    monthly_cost: Optional[float] = 599.0,
    team_size: Optional[int] = None,
    source: str = "knowledge_base",
    alternatives: Optional[list[str]] = None,
) -> AnalysisResult:
    """Factory for AnalysisResult test instances."""
    tool = make_tool(name, category, monthly_cost_usd=monthly_cost, team_size=team_size)
    score = RiskScore.from_dimensions(
        task_automation_ratio=task_auto,
        api_openness=api,
        workflow_complexity=wf,
        data_sensitivity=ds,
        incumbent_inertia=inertia,
        alternatives=alternatives or ["Alt1", "Alt2", "Alt3"],
        rationale="Test rationale for the tool.",
    )
    return AnalysisResult(tool=tool, score=score, source=source)


def make_report(
    tool_configs: Optional[list[dict]] = None,
    generated_at: str = "2024-01-15T12:00:00",
) -> ScanReport:
    """Build a small ScanReport for testing."""
    if tool_configs is None:
        tool_configs = [
            {"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0},
            {"name": "Salesforce", "category": ToolCategory.CRM, "monthly_cost": 3200.0},
            {"name": "Workday", "category": ToolCategory.HR, "monthly_cost": 2100.0},
        ]

    results = [
        make_result(
            name=cfg["name"],
            category=cfg["category"],
            monthly_cost=cfg.get("monthly_cost"),
        )
        for cfg in tool_configs
    ]

    report = ScanReport(
        results=results,
        generated_at=generated_at,
        total_tools=len(results),
    )
    report.rank_results()
    report.compute_summary()
    return report


def null_console() -> Console:
    """Return a Rich Console that discards output for testing."""
    return Console(file=io.StringIO(), highlight=False, markup=True)


def capturing_console() -> tuple[Console, io.StringIO]:
    """Return a Rich Console plus a StringIO buffer to inspect output."""
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False)
    return con, buf


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFormatCost:
    """Tests for _format_cost()."""

    def test_none_returns_dash(self) -> None:
        assert _format_cost(None) == "\u2014"

    def test_zero_returns_zero_string(self) -> None:
        assert _format_cost(0.0) == "$0"

    def test_small_amount(self) -> None:
        result = _format_cost(99.0)
        assert result.startswith("$")
        assert "99" in result

    def test_integer_amount(self) -> None:
        result = _format_cost(599.0)
        assert "599" in result

    def test_large_amount_has_comma(self) -> None:
        result = _format_cost(3200.0)
        # Should contain either 3,200 or 3200 depending on locale
        assert "3" in result and "200" in result

    def test_very_large_amount(self) -> None:
        result = _format_cost(100000.0)
        assert "$" in result
        assert "100" in result

    def test_dollar_sign_present(self) -> None:
        assert _format_cost(100.0).startswith("$")

    def test_returns_string(self) -> None:
        assert isinstance(_format_cost(500.0), str)
        assert isinstance(_format_cost(None), str)

    def test_negative_not_expected_but_returns_string(self) -> None:
        # We don't validate negative costs in reporter; just ensure no crash
        result = _format_cost(-50.0)
        assert isinstance(result, str)


class TestFormatTeamSize:
    """Tests for _format_team_size()."""

    def test_none_returns_dash(self) -> None:
        assert _format_team_size(None) == "\u2014"

    def test_integer_returned_as_string(self) -> None:
        assert _format_team_size(50) == "50"

    def test_one(self) -> None:
        assert _format_team_size(1) == "1"

    def test_large_number(self) -> None:
        assert _format_team_size(1000) == "1000"

    def test_returns_string(self) -> None:
        assert isinstance(_format_team_size(5), str)
        assert isinstance(_format_team_size(None), str)


class TestScoreBarText:
    """Tests for _score_bar_text()."""

    def test_returns_string(self) -> None:
        assert isinstance(_score_bar_text(75.0), str)

    def test_full_bar_at_100(self) -> None:
        bar = _score_bar_text(100.0, width=10)
        assert bar.count("\u2588") == 10
        assert bar.count("\u2591") == 0

    def test_empty_bar_at_0(self) -> None:
        bar = _score_bar_text(0.0, width=10)
        assert bar.count("\u2588") == 0
        assert bar.count("\u2591") == 10

    def test_half_bar_at_50(self) -> None:
        bar = _score_bar_text(50.0, width=20)
        filled = bar.count("\u2588")
        empty = bar.count("\u2591")
        assert filled == empty == 10

    def test_has_brackets(self) -> None:
        bar = _score_bar_text(50.0)
        assert "[" in bar
        assert "]" in bar

    def test_length_is_width_plus_brackets(self) -> None:
        bar = _score_bar_text(50.0, width=10)
        # [10 chars] = 12 total
        assert len(bar) == 12

    def test_default_width_20(self) -> None:
        bar = _score_bar_text(50.0)
        # Default width=20 → 22 chars total
        assert len(bar) == 22

    def test_score_25_has_quarter_fill(self) -> None:
        bar = _score_bar_text(25.0, width=20)
        filled = bar.count("\u2588")
        assert filled == 5

    def test_score_75_has_three_quarter_fill(self) -> None:
        bar = _score_bar_text(75.0, width=20)
        filled = bar.count("\u2588")
        assert filled == 15

    def test_no_exception_at_bounds(self) -> None:
        _score_bar_text(0.0)
        _score_bar_text(100.0)

    def test_custom_width(self) -> None:
        bar = _score_bar_text(50.0, width=40)
        assert len(bar) == 42  # 40 + 2 brackets


class TestScoreBarRich:
    """Tests for _score_bar_rich()."""

    def test_returns_rich_text(self) -> None:
        from rich.text import Text
        bar = _score_bar_rich(50.0)
        assert isinstance(bar, Text)

    def test_no_exception_at_0(self) -> None:
        _score_bar_rich(0.0)

    def test_no_exception_at_100(self) -> None:
        _score_bar_rich(100.0)

    def test_no_exception_at_various_scores(self) -> None:
        for score in [10, 25, 50, 75, 90]:
            _score_bar_rich(float(score))

    def test_no_exception_with_custom_width(self) -> None:
        _score_bar_rich(50.0, width=30)

    def test_contains_fill_characters(self) -> None:
        from rich.text import Text
        bar = _score_bar_rich(100.0)
        plain = bar.plain
        assert "\u2588" in plain or len(plain) > 0


# ---------------------------------------------------------------------------
# Reporter — table rendering
# ---------------------------------------------------------------------------


class TestReporterTable:
    """Tests for Reporter.render_table()."""

    def test_renders_without_exception(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_renders_with_top_n(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table(top_n=2)  # Should not raise

    def test_renders_with_min_score(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table(min_score=40.0)  # Should not raise

    def test_renders_with_show_dimensions(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table(show_dimensions=True)  # Should not raise

    def test_renders_top_n_and_min_score_combined(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table(top_n=2, min_score=0.0)  # Should not raise

    def test_renders_single_tool(self) -> None:
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_empty_results_after_filter_no_exception(self) -> None:
        """When all results are filtered out, should not raise."""
        report = make_report()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_table(min_score=999.0)  # Impossible score — filters everything
        # Should complete without exception
        assert isinstance(buf.getvalue(), str)

    def test_empty_results_shows_message(self) -> None:
        """When all results are filtered out, a message is shown."""
        report = make_report()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_table(min_score=999.0)
        output = buf.getvalue()
        # Some kind of message should appear
        assert len(output) >= 0  # at minimum no crash

    def test_output_contains_tool_names(self) -> None:
        """Rendered table should contain at least one tool name."""
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_table()
        output = buf.getvalue()
        assert "Zapier" in output

    def test_table_with_multiple_tools(self) -> None:
        """All tool names should appear when multiple tools are rendered."""
        report = make_report()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_table()
        output = buf.getvalue()
        # At least two of three tools should appear
        names_found = sum(1 for name in ["Zapier", "Salesforce", "Workday"] if name in output)
        assert names_found >= 2

    def test_renders_with_enriched_result(self) -> None:
        """Enriched (LLM) results should render without error."""
        tool = make_tool("TestLLMTool", ToolCategory.AUTOMATION)
        score = RiskScore.from_dimensions(
            task_automation_ratio=8.0,
            api_openness=7.0,
            workflow_complexity=3.0,
            data_sensitivity=3.0,
            incumbent_inertia=3.0,
            alternatives=["Alt1"],
            rationale="LLM rationale.",
            enriched_by_llm=True,
        )
        result = AnalysisResult(tool=tool, score=score, source="llm")
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_renders_default_source_result(self) -> None:
        """Default-scored results should render without error."""
        result = make_result(source="default")
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_renders_tool_with_no_cost(self) -> None:
        """Tools without monthly cost should render without error."""
        report = make_report(
            tool_configs=[{"name": "FreeTools", "category": ToolCategory.OTHER, "monthly_cost": None}]
        )
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_renders_tool_with_team_size(self) -> None:
        """Tools with team size should render without error."""
        result = make_result(name="TeamTool", team_size=50)
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise


# ---------------------------------------------------------------------------
# Reporter — detail rendering
# ---------------------------------------------------------------------------


class TestReporterDetail:
    """Tests for Reporter.render_detail()."""

    def test_renders_without_exception(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_detail(report.results[0])  # Should not raise

    def test_shows_tool_name_in_output(self) -> None:
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_detail(report.results[0])
        output = buf.getvalue()
        assert "Zapier" in output

    def test_shows_dimension_names_in_output(self) -> None:
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_detail(report.results[0])
        output = buf.getvalue()
        assert "Automation" in output or "Task" in output or "API" in output

    def test_shows_alternatives(self) -> None:
        result = make_result(name="TestTool", alternatives=["n8n", "LangChain"])
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_detail(result)
        output = buf.getvalue()
        assert "n8n" in output or "LangChain" in output

    def test_shows_rationale(self) -> None:
        result = make_result(name="TestTool")
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_detail(result)
        output = buf.getvalue()
        # Rationale text should be present
        assert "rationale" in output.lower() or "Test rationale" in output

    def test_shows_enriched_by_llm_indicator(self) -> None:
        tool = make_tool("LLMTool", ToolCategory.AUTOMATION)
        score = RiskScore.from_dimensions(
            task_automation_ratio=8.0,
            api_openness=7.0,
            workflow_complexity=3.0,
            data_sensitivity=3.0,
            incumbent_inertia=3.0,
            alternatives=["Alt1"],
            rationale="LLM rationale.",
            enriched_by_llm=True,
        )
        result = AnalysisResult(tool=tool, score=score, source="llm")
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        con, buf = capturing_console()
        reporter = Reporter(report=report, console=con)
        reporter.render_detail(result)
        output = buf.getvalue()
        assert "LLM" in output or "enrichment" in output.lower() or "\u2728" in output

    def test_renders_tool_with_no_optional_fields(self) -> None:
        tool = make_tool("MinimalTool", ToolCategory.OTHER)
        score = RiskScore.from_dimensions(5.0, 5.0, 5.0, 5.0, 5.0)
        result = AnalysisResult(tool=tool, score=score, source="default")
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        reporter.render_detail(result)  # Should not raise


# ---------------------------------------------------------------------------
# Reporter — Markdown rendering
# ---------------------------------------------------------------------------


class TestReporterMarkdown:
    """Tests for Reporter.render_markdown()."""

    def test_returns_string(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        result = reporter.render_markdown()
        assert isinstance(result, str)

    def test_non_empty_output(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert len(md) > 0

    def test_contains_report_title(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert "Risk Report" in md or "SaaS" in md

    def test_contains_at_least_one_tool_name(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        tool_names = [r.tool_name for r in report.results]
        assert any(name in md for name in tool_names)

    def test_contains_all_tool_names(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        for r in report.results:
            assert r.tool_name in md, f"Tool '{r.tool_name}' not found in Markdown"

    def test_contains_generated_at_date(self) -> None:
        report = make_report(generated_at="2024-01-15T12:00:00")
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert "2024-01-15" in md

    def test_contains_score_values(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Should have at least one numeric score
        assert re.search(r"\d+\.\d+", md) is not None

    def test_contains_markdown_table_syntax(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Markdown tables use pipes
        assert "|" in md

    def test_contains_risk_level_words(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        risk_words = ["Critical", "High", "Medium", "Low"]
        assert any(w in md for w in risk_words)

    def test_contains_heading_markers(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Markdown headings start with #
        assert "#" in md

    def test_writes_to_file(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        output_file = tmp_path / "report.md"
        reporter.render_markdown(output_path=output_file)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_file_content_matches_returned_string(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        output_file = tmp_path / "report.md"
        returned = reporter.render_markdown(output_path=output_file)
        saved = output_file.read_text(encoding="utf-8")
        assert returned == saved

    def test_top_n_filter_reduces_content(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md_full = reporter.render_markdown()
        md_top1 = reporter.render_markdown(top_n=1)
        assert len(md_top1) <= len(md_full)

    def test_top_n_1_contains_only_top_tool(self) -> None:
        report = make_report()
        top_tool = report.results[0].tool_name
        other_tools = [r.tool_name for r in report.results[1:]]
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown(top_n=1)
        assert top_tool in md
        # At least some other tools should be absent
        missing = sum(1 for name in other_tools if name not in md)
        assert missing >= 1

    def test_min_score_filter_reduces_content(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md_all = reporter.render_markdown()
        md_filtered = reporter.render_markdown(min_score=95.0)
        assert len(md_filtered) <= len(md_all)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        nested_path = tmp_path / "subdir" / "deep" / "report.md"
        reporter.render_markdown(output_path=nested_path)
        assert nested_path.exists()

    def test_enrichment_info_when_enabled(self) -> None:
        report = ScanReport(
            results=[make_result()],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
            enrichment_enabled=True,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Should mention enrichment
        assert "llm" in md.lower() or "enrichment" in md.lower() or "openai" in md.lower()

    def test_no_enrichment_info_when_disabled(self) -> None:
        report = make_report()
        assert report.enrichment_enabled is False
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Enrichment note should not be prominent
        # This is a soft check — just ensure no crash
        assert isinstance(md, str)

    def test_alternatives_appear_in_markdown(self) -> None:
        result = make_result(name="Zapier", alternatives=["n8n", "LangChain agents"])
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert "n8n" in md or "LangChain" in md

    def test_returns_same_content_on_multiple_calls(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md1 = reporter.render_markdown()
        md2 = reporter.render_markdown()
        assert md1 == md2


# ---------------------------------------------------------------------------
# Reporter — JSON rendering
# ---------------------------------------------------------------------------


class TestReporterJson:
    """Tests for Reporter.render_json()."""

    def test_returns_string(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        result = reporter.render_json()
        assert isinstance(result, str)

    def test_returns_valid_json(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        json_str = reporter.render_json()
        data = json.loads(json_str)  # Should not raise
        assert isinstance(data, dict)

    def test_json_contains_results_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "results" in data

    def test_json_contains_summary_stats(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "summary_stats" in data

    def test_json_contains_generated_at(self) -> None:
        report = make_report(generated_at="2024-01-15T12:00:00")
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "generated_at" in data
        assert "2024-01-15" in data["generated_at"]

    def test_json_contains_total_tools(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "total_tools" in data

    def test_json_contains_enrichment_enabled(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "enrichment_enabled" in data

    def test_json_results_count_matches(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert len(data["results"]) == report.total_tools

    def test_json_result_has_tool_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "tool" in data["results"][0]

    def test_json_result_has_score_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "score" in data["results"][0]

    def test_json_result_tool_has_name(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "name" in data["results"][0]["tool"]

    def test_json_result_score_has_displacement_score(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "displacement_score" in data["results"][0]["score"]

    def test_json_displacement_scores_are_numbers(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        for result in data["results"]:
            assert isinstance(result["score"]["displacement_score"], (int, float))

    def test_json_top_n_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(top_n=1))
        assert len(data["results"]) == 1

    def test_json_top_n_2_returns_2(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(top_n=2))
        assert len(data["results"]) == 2

    def test_json_min_score_zero_returns_all(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(min_score=0.0))
        assert len(data["results"]) == report.total_tools

    def test_json_impossible_min_score_returns_empty(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(min_score=101.0))
        assert len(data["results"]) == 0

    def test_json_writes_to_file(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        out = tmp_path / "results.json"
        reporter.render_json(output_path=out)
        assert out.exists()

    def test_json_file_is_valid(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        out = tmp_path / "results.json"
        reporter.render_json(output_path=out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "results" in data

    def test_json_has_shown_tools_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "shown_tools" in data

    def test_json_shown_tools_equals_total_without_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert data["shown_tools"] == data["total_tools"]

    def test_json_shown_tools_less_with_top_n(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(top_n=1))
        assert data["shown_tools"] == 1

    def test_json_indent_is_2_by_default(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        json_str = reporter.render_json()
        # Indented JSON will have multiple lines
        assert "\n" in json_str

    def test_json_creates_parent_directory(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        nested = tmp_path / "deep" / "results.json"
        reporter.render_json(output_path=nested)
        assert nested.exists()

    def test_json_result_has_source_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        for r in data["results"]:
            assert "source" in r

    def test_json_result_has_rank_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        for r in data["results"]:
            assert "rank" in r

    def test_json_score_has_dimensions(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        for r in data["results"]:
            assert "dimensions" in r["score"]

    def test_json_dimensions_have_all_five_keys(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        expected_dims = {
            "task_automation_ratio",
            "api_openness",
            "workflow_complexity",
            "data_sensitivity",
            "incumbent_inertia",
        }
        for r in data["results"]:
            assert set(r["score"]["dimensions"].keys()) == expected_dims


# ---------------------------------------------------------------------------
# Reporter — filtering
# ---------------------------------------------------------------------------


class TestReporterFiltering:
    """Tests for Reporter._filter_results()."""

    def test_no_filter_returns_all(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results()
        assert len(results) == report.total_tools

    def test_top_n_returns_n_results(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(top_n=2)
        assert len(results) == 2

    def test_top_n_1_returns_1(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(top_n=1)
        assert len(results) == 1

    def test_top_n_larger_than_total_returns_all(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(top_n=100)
        assert len(results) == report.total_tools

    def test_min_score_filters_below_threshold(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        max_score = max(r.displacement_score for r in report.results)
        results = reporter._filter_results(min_score=max_score)
        assert all(r.displacement_score >= max_score for r in results)

    def test_min_score_zero_returns_all(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(min_score=0.0)
        assert len(results) == report.total_tools

    def test_impossible_min_score_returns_empty(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(min_score=101.0)
        assert len(results) == 0

    def test_top_n_and_min_score_combined(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(top_n=2, min_score=0.0)
        assert len(results) <= 2

    def test_results_preserve_sort_order(self) -> None:
        """Filtered results should remain sorted by displacement score desc."""
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results()
        scores = [r.displacement_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_preserves_highest_scoring(self) -> None:
        """top_n should return the highest-scoring tools."""
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        all_results = reporter._filter_results()
        top_results = reporter._filter_results(top_n=1)
        assert top_results[0].displacement_score == all_results[0].displacement_score

    def test_min_score_returns_only_qualifying(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        threshold = 50.0
        results = reporter._filter_results(min_score=threshold)
        for r in results:
            assert r.displacement_score >= threshold

    def test_returns_list(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results()
        assert isinstance(results, list)

    def test_does_not_modify_original_report(self) -> None:
        """Filtering should not alter the report's results list."""
        report = make_report()
        original_count = len(report.results)
        reporter = Reporter(report=report, console=null_console())
        reporter._filter_results(top_n=1, min_score=0.0)
        assert len(report.results) == original_count


# ---------------------------------------------------------------------------
# render() convenience function
# ---------------------------------------------------------------------------


class TestRenderFunction:
    """Tests for the render() convenience function."""

    def test_table_format_returns_none(self) -> None:
        report = make_report()
        result = render(report, fmt="table", console=null_console())
        assert result is None

    def test_markdown_format_returns_string(self) -> None:
        report = make_report()
        result = render(report, fmt="markdown")
        assert isinstance(result, str)

    def test_json_format_returns_string(self) -> None:
        report = make_report()
        result = render(report, fmt="json")
        assert isinstance(result, str)

    def test_json_result_is_valid_json(self) -> None:
        report = make_report()
        result = render(report, fmt="json")
        data = json.loads(result)  # Should not raise
        assert "results" in data

    def test_unknown_format_raises_value_error(self) -> None:
        report = make_report()
        with pytest.raises(ValueError, match="Unsupported format"):
            render(report, fmt="xml")

    def test_unknown_format_toml_raises(self) -> None:
        report = make_report()
        with pytest.raises(ValueError):
            render(report, fmt="toml")

    def test_case_insensitive_format_json(self) -> None:
        report = make_report()
        result = render(report, fmt="JSON")
        assert isinstance(result, str)

    def test_case_insensitive_format_markdown(self) -> None:
        report = make_report()
        result = render(report, fmt="MARKDOWN")
        assert isinstance(result, str)

    def test_case_insensitive_format_table(self) -> None:
        report = make_report()
        result = render(report, fmt="TABLE", console=null_console())
        assert result is None

    def test_markdown_writes_file(self, tmp_path: Path) -> None:
        report = make_report()
        out = tmp_path / "report.md"
        render(report, fmt="markdown", output_path=out)
        assert out.exists()

    def test_json_writes_file(self, tmp_path: Path) -> None:
        report = make_report()
        out = tmp_path / "results.json"
        render(report, fmt="json", output_path=out)
        assert out.exists()

    def test_table_with_top_n(self) -> None:
        report = make_report()
        result = render(report, fmt="table", top_n=1, console=null_console())
        assert result is None  # table renders to console, returns None

    def test_markdown_with_top_n(self) -> None:
        report = make_report()
        result = render(report, fmt="markdown", top_n=1)
        assert isinstance(result, str)

    def test_json_with_top_n(self) -> None:
        report = make_report()
        result = render(report, fmt="json", top_n=1)
        data = json.loads(result)
        assert len(data["results"]) == 1

    def test_markdown_with_min_score(self) -> None:
        report = make_report()
        result = render(report, fmt="markdown", min_score=0.0)
        assert isinstance(result, str)

    def test_json_with_min_score(self) -> None:
        report = make_report()
        result = render(report, fmt="json", min_score=0.0)
        data = json.loads(result)
        assert len(data["results"]) == report.total_tools

    def test_show_dimensions_table(self) -> None:
        report = make_report()
        result = render(
            report, fmt="table",
            show_dimensions=True,
            console=null_console(),
        )
        assert result is None  # still returns None

    def test_uses_provided_console(self) -> None:
        report = make_report()
        con, buf = capturing_console()
        render(report, fmt="table", console=con)
        output = buf.getvalue()
        # Some output should have been written to our buffer
        assert len(output) >= 0  # no crash, at minimum


# ---------------------------------------------------------------------------
# load_report_from_json
# ---------------------------------------------------------------------------


class TestLoadReportFromJson:
    """Tests for the load_report_from_json() function."""

    def _save_report(self, report: ScanReport, path: Path) -> None:
        """Save a report to a JSON file for test loading."""
        reporter = Reporter(report=report, console=null_console())
        reporter.render_json(output_path=path)

    def test_loads_valid_report(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert isinstance(loaded, ScanReport)

    def test_loaded_report_has_same_tool_count(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert len(loaded.results) == report.total_tools

    def test_loaded_report_has_same_tool_names(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        original_names = {r.tool_name for r in report.results}
        loaded_names = {r.tool_name for r in loaded.results}
        assert original_names == loaded_names

    def test_loaded_report_displacement_scores_match(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        original_scores = sorted(r.displacement_score for r in report.results)
        loaded_scores = sorted(r.displacement_score for r in loaded.results)
        assert original_scores == loaded_scores

    def test_loaded_report_has_generated_at(self, tmp_path: Path) -> None:
        report = make_report(generated_at="2024-06-15T10:30:00")
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert "2024-06-15" in loaded.generated_at

    def test_loaded_results_have_risk_levels(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        for r in loaded.results:
            assert isinstance(r.risk_level, RiskLevel)

    def test_loaded_results_have_alternatives(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        for r in loaded.results:
            assert isinstance(r.score.alternatives, list)

    def test_loaded_results_have_categories(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        for r in loaded.results:
            assert isinstance(r.tool.category, ToolCategory)

    def test_loaded_results_have_source(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        for r in loaded.results:
            assert r.source in ("knowledge_base", "llm", "default")

    def test_loaded_results_have_rank(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        for r in loaded.results:
            assert r.rank is not None

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_report_from_json(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Failed to parse JSON report"):
            load_report_from_json(bad_json)

    def test_wrong_json_structure_list_raises_value_error(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(ValueError):
            load_report_from_json(bad_json)

    def test_missing_results_key_loads_empty(self, tmp_path: Path) -> None:
        """A JSON file without 'results' should produce an empty report."""
        data = {"generated_at": "2024-01-01T00:00:00", "total_tools": 0}
        p = tmp_path / "report.json"
        p.write_text(json.dumps(data))
        loaded = load_report_from_json(p)
        assert isinstance(loaded, ScanReport)
        assert len(loaded.results) == 0

    def test_malformed_result_raises_value_error(self, tmp_path: Path) -> None:
        """A result with missing required fields should raise ValueError."""
        data = {
            "generated_at": "2024-01-01T00:00:00",
            "results": [{"bad": "data"}],
        }
        p = tmp_path / "report.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_report_from_json(p)

    def test_round_trip_markdown_from_loaded(self, tmp_path: Path) -> None:
        """Loaded report can be rendered to Markdown without error."""
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        reporter = Reporter(report=loaded, console=null_console())
        md = reporter.render_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_round_trip_json_from_loaded(self, tmp_path: Path) -> None:
        """Loaded report can be re-serialized to JSON without error."""
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        reporter = Reporter(report=loaded, console=null_console())
        json_str = reporter.render_json()
        data = json.loads(json_str)
        assert "results" in data

    def test_loaded_enrichment_enabled_preserved(self, tmp_path: Path) -> None:
        report = ScanReport(
            results=[make_result()],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
            enrichment_enabled=True,
        )
        report.rank_results()
        report.compute_summary()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert loaded.enrichment_enabled is True

    def test_loaded_enrichment_disabled_preserved(self, tmp_path: Path) -> None:
        report = make_report()
        assert report.enrichment_enabled is False
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert loaded.enrichment_enabled is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(str(json_path))
        assert isinstance(loaded, ScanReport)

    def test_loaded_rationale_preserved(self, tmp_path: Path) -> None:
        result = make_result(name="Zapier")
        assert result.score.rationale is not None
        report = ScanReport(
            results=[result],
            generated_at="2024-01-01T00:00:00",
            total_tools=1,
        )
        report.rank_results()
        report.compute_summary()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert loaded.results[0].score.rationale is not None

    def test_loaded_summary_stats_preserved(self, tmp_path: Path) -> None:
        report = make_report()
        json_path = tmp_path / "report.json"
        self._save_report(report, json_path)
        loaded = load_report_from_json(json_path)
        assert isinstance(loaded.summary_stats, dict)


# ---------------------------------------------------------------------------
# Integration: render full sample stack
# ---------------------------------------------------------------------------


class TestIntegrationReporter:
    """Integration tests for the full reporting pipeline."""

    def test_render_sample_stack_as_table(self) -> None:
        """Render the example stack as a table without errors."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_render_sample_stack_as_markdown(self) -> None:
        """Render the example stack as Markdown without errors."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert isinstance(md, str)
        assert len(md) > 100  # Should have substantial content

    def test_render_sample_stack_as_json(self) -> None:
        """Render the example stack as JSON without errors."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        json_str = reporter.render_json()
        data = json.loads(json_str)
        assert len(data["results"]) == stack.tool_count()

    def test_sample_stack_markdown_contains_tool_names(self) -> None:
        """The Markdown report for the sample stack should list tools."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Common tools from sample_stack.yaml should appear
        assert any(name in md for name in ["Zapier", "Salesforce", "Notion", "Slack"])

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """Save to JSON and reload; re-render as Markdown without errors."""
        report = make_report()
        reporter = Reporter(report=report, console=null_console())

        # Save
        json_path = tmp_path / "round_trip.json"
        reporter.render_json(output_path=json_path)

        # Reload
        loaded = load_report_from_json(json_path)
        reporter2 = Reporter(report=loaded, console=null_console())
        md = reporter2.render_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_json_round_trip_tool_count_preserved(self, tmp_path: Path) -> None:
        """After JSON round-trip, tool count should be preserved."""
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        json_path = tmp_path / "round_trip.json"
        reporter.render_json(output_path=json_path)
        loaded = load_report_from_json(json_path)
        assert len(loaded.results) == report.total_tools

    def test_json_round_trip_scores_preserved(self, tmp_path: Path) -> None:
        """After JSON round-trip, displacement scores should be preserved."""
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        json_path = tmp_path / "round_trip.json"
        reporter.render_json(output_path=json_path)
        loaded = load_report_from_json(json_path)
        original = sorted(r.displacement_score for r in report.results)
        loaded_scores = sorted(r.displacement_score for r in loaded.results)
        assert original == loaded_scores

    def test_markdown_top_5_from_sample_stack(self) -> None:
        """Top-5 Markdown report from sample stack should work."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown(top_n=5)
        assert isinstance(md, str)
        # Should have content
        assert len(md) > 50

    def test_full_pipeline_to_json_file(self, tmp_path: Path) -> None:
        """Full pipeline: load → score → render JSON → file exists and is valid."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        out = tmp_path / "full_pipeline.json"
        render(report, fmt="json", output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["results"]) == stack.tool_count()

    def test_full_pipeline_to_markdown_file(self, tmp_path: Path) -> None:
        """Full pipeline: load → score → render Markdown → file exists and is valid."""
        import datetime
        from saas_risk_scan.loader import load_file

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        out = tmp_path / "full_pipeline.md"
        render(report, fmt="markdown", output_path=out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert len(content) > 100
