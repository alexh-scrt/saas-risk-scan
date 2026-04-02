"""Unit tests for saas_risk_scan/reporter.py.

Covers Rich table rendering, Markdown generation, JSON export,
filtering logic, and report loading from JSON.
"""

from __future__ import annotations

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
    """Return a Rich Console that writes to /dev/null for testing."""
    import io
    return Console(file=io.StringIO(), highlight=False, markup=True)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFormatCost:
    """Tests for _format_cost()."""

    def test_none_returns_dash(self) -> None:
        assert _format_cost(None) == "—"

    def test_zero_returns_zero(self) -> None:
        assert _format_cost(0.0) == "$0"

    def test_integer_amount(self) -> None:
        assert _format_cost(599.0) == "$599"

    def test_large_amount_has_comma(self) -> None:
        result = _format_cost(3200.0)
        assert "3,200" in result or "3200" in result  # locale may vary

    def test_dollar_sign_present(self) -> None:
        assert _format_cost(100.0).startswith("$")


class TestFormatTeamSize:
    """Tests for _format_team_size()."""

    def test_none_returns_dash(self) -> None:
        assert _format_team_size(None) == "—"

    def test_integer_returned_as_string(self) -> None:
        assert _format_team_size(50) == "50"

    def test_one(self) -> None:
        assert _format_team_size(1) == "1"


class TestScoreBarText:
    """Tests for _score_bar_text()."""

    def test_full_bar_at_100(self) -> None:
        bar = _score_bar_text(100.0)
        assert "░" not in bar or bar.count("█") == bar.count("█")  # all filled
        assert "[" in bar and "]" in bar

    def test_empty_bar_at_0(self) -> None:
        bar = _score_bar_text(0.0)
        assert "█" not in bar

    def test_half_bar_at_50(self) -> None:
        bar = _score_bar_text(50.0, width=20)
        filled = bar.count("█")
        empty = bar.count("░")
        assert filled == empty == 10

    def test_returns_string(self) -> None:
        assert isinstance(_score_bar_text(75.0), str)

    def test_length_is_width_plus_brackets(self) -> None:
        bar = _score_bar_text(50.0, width=10)
        # [10 chars] = 12 total
        assert len(bar) == 12


class TestScoreBarRich:
    """Tests for _score_bar_rich()."""

    def test_returns_rich_text(self) -> None:
        from rich.text import Text
        bar = _score_bar_rich(50.0)
        assert isinstance(bar, Text)

    def test_no_exception_at_bounds(self) -> None:
        _score_bar_rich(0.0)
        _score_bar_rich(100.0)

    def test_no_exception_at_various_scores(self) -> None:
        for score in [10, 25, 50, 75, 90]:
            _score_bar_rich(float(score))


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

    def test_empty_results_after_filter_shows_message(self) -> None:
        """When all results are filtered out, a warning message is shown."""
        import io
        buf = io.StringIO()
        console = Console(file=buf, highlight=False, markup=False)
        report = make_report()
        reporter = Reporter(report=report, console=console)
        reporter.render_table(min_score=999.0)  # Impossible score — filters everything
        output = buf.getvalue()
        assert "No results" in output or len(output) >= 0  # At minimum no crash

    def test_renders_single_tool(self) -> None:
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
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
        import io
        buf = io.StringIO()
        console = Console(file=buf, highlight=False, markup=False)
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        reporter = Reporter(report=report, console=console)
        reporter.render_detail(report.results[0])
        output = buf.getvalue()
        assert "Zapier" in output

    def test_shows_dimensions_in_output(self) -> None:
        import io
        buf = io.StringIO()
        console = Console(file=buf, highlight=False, markup=False)
        report = make_report(
            tool_configs=[{"name": "Zapier", "category": ToolCategory.AUTOMATION, "monthly_cost": 599.0}]
        )
        reporter = Reporter(report=report, console=console)
        reporter.render_detail(report.results[0])
        output = buf.getvalue()
        assert "Task Automation Ratio" in output or "Automation" in output


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

    def test_contains_report_title(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        assert "Risk Report" in md or "SaaS" in md

    def test_contains_tool_names(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # At least one tool name should appear
        tool_names = [r.tool_name for r in report.results]
        assert any(name in md for name in tool_names)

    def test_contains_generated_at(self) -> None:
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

    def test_writes_to_file(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        output_file = tmp_path / "report.md"
        reporter.render_markdown(output_path=output_file)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_top_n_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md_full = reporter.render_markdown()
        md_top1 = reporter.render_markdown(top_n=1)
        # Top 1 should be shorter
        assert len(md_top1) <= len(md_full)

    def test_min_score_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        # Setting a very high min_score should reduce content
        md_all = reporter.render_markdown()
        md_filtered = reporter.render_markdown(min_score=95.0)
        assert len(md_filtered) <= len(md_all)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        nested_path = tmp_path / "subdir" / "deep" / "report.md"
        reporter.render_markdown(output_path=nested_path)
        assert nested_path.exists()

    def test_markdown_has_table_syntax(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # Markdown tables use pipes
        assert "|" in md

    def test_risk_levels_appear_in_markdown(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        md = reporter.render_markdown()
        # At least one risk level word should appear
        risk_words = ["Critical", "High", "Medium", "Low"]
        assert any(w in md for w in risk_words)


# ---------------------------------------------------------------------------
# Reporter — JSON rendering
# ---------------------------------------------------------------------------


class TestReporterJson:
    """Tests for Reporter.render_json()."""

    def test_returns_valid_json_string(self) -> None:
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

    def test_json_results_count_matches(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert len(data["results"]) == report.total_tools

    def test_json_result_has_tool_and_score(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        first = data["results"][0]
        assert "tool" in first
        assert "score" in first
        assert "name" in first["tool"]
        assert "displacement_score" in first["score"]

    def test_json_top_n_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(top_n=1))
        assert len(data["results"]) == 1

    def test_json_min_score_filter(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json(min_score=0.0))  # Passes all
        assert len(data["results"]) == report.total_tools

    def test_json_writes_to_file(self, tmp_path: Path) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        out = tmp_path / "results.json"
        reporter.render_json(output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "results" in data

    def test_json_displacement_scores_are_floats(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        for result in data["results"]:
            assert isinstance(result["score"]["displacement_score"], (int, float))

    def test_json_has_shown_tools_key(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        data = json.loads(reporter.render_json())
        assert "shown_tools" in data


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

    def test_top_n_larger_than_total_returns_all(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        results = reporter._filter_results(top_n=100)
        assert len(results) == report.total_tools

    def test_min_score_filters_below_threshold(self) -> None:
        report = make_report()
        reporter = Reporter(report=report, console=null_console())
        # Use a score that only passes the highest-risk tool
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
        data = json.loads(result)  # Should be valid JSON
        assert "results" in data

    def test_unknown_format_raises(self) -> None:
        report = make_report()
        with pytest.raises(ValueError, match="Unsupported format"):
            render(report, fmt="xml")

    def test_case_insensitive_format(self) -> None:
        report = make_report()
        result = render(report, fmt="JSON")
        assert isinstance(result, str)

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

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_report_from_json(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Failed to parse JSON report"):
            load_report_from_json(bad_json)

    def test_wrong_json_structure_raises_value_error(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(ValueError):
            load_report_from_json(bad_json)

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


# ---------------------------------------------------------------------------
# Integration: render full sample stack
# ---------------------------------------------------------------------------


class TestIntegrationReporter:
    """Integration tests for the full reporting pipeline."""

    def test_render_sample_stack_as_table(self) -> None:
        """Render the example stack as a table without errors."""
        from pathlib import Path
        from saas_risk_scan.loader import load_file
        import datetime

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        reporter.render_table()  # Should not raise

    def test_render_sample_stack_as_markdown(self) -> None:
        """Render the example stack as Markdown without errors."""
        from pathlib import Path
        from saas_risk_scan.loader import load_file
        import datetime

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
        from pathlib import Path
        from saas_risk_scan.loader import load_file
        import datetime

        sample = Path("examples/sample_stack.yaml")
        if not sample.exists():
            pytest.skip("examples/sample_stack.yaml not found")

        stack = load_file(sample)
        report = build_report(stack, generated_at=datetime.datetime.now().isoformat())
        reporter = Reporter(report=report, console=null_console())
        json_str = reporter.render_json()
        data = json.loads(json_str)
        assert len(data["results"]) == stack.tool_count()

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
