"""Report rendering module for SaaS Risk Scan.

This module renders ScanReport objects into three output formats:
    1. Rich terminal table  — colorized, human-readable table for the console.
    2. Markdown file        — Jinja2-templated Markdown report for documentation.
    3. JSON export          — structured JSON for programmatic consumption.

All rendering is done via the Reporter class, which accepts a ScanReport and
provides methods for each output format. The ``render`` convenience function
wraps the class for simple one-shot usage.

Jinja2 templates are loaded from the ``templates/`` directory relative to the
project root, with a fallback to the package directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from saas_risk_scan.models import (
    AnalysisResult,
    RiskLevel,
    ScanReport,
    ReplacementTimeline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Risk level display configuration
_RISK_COLORS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "bold red",
    RiskLevel.HIGH: "bold yellow",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.LOW: "green",
}

_RISK_EMOJIS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "🔴",
    RiskLevel.HIGH: "🟠",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.LOW: "🟢",
}

_TIMELINE_COLORS: dict[ReplacementTimeline, str] = {
    ReplacementTimeline.NEAR: "bold red",
    ReplacementTimeline.MID: "yellow",
    ReplacementTimeline.LONG: "cyan",
    ReplacementTimeline.UNLIKELY: "green",
}

# Score bar width for terminal display
_SCORE_BAR_WIDTH = 20

# Maximum alternatives to show in the table
_MAX_ALTS_IN_TABLE = 3

# Template filename
_TEMPLATE_FILENAME = "report.md.j2"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def _find_template_dirs() -> list[Path]:
    """Return candidate directories for Jinja2 template loading.

    Searches in order:
    1. ``templates/`` relative to the current working directory.
    2. ``templates/`` relative to the package source directory.
    3. The package source directory itself (fallback).

    Returns:
        List of existing Path objects to search for templates.
    """
    candidates: list[Path] = []

    # CWD-relative templates dir (standard project layout)
    cwd_templates = Path.cwd() / "templates"
    if cwd_templates.is_dir():
        candidates.append(cwd_templates)

    # Package-relative templates dir
    package_dir = Path(__file__).parent
    pkg_templates = package_dir.parent / "templates"
    if pkg_templates.is_dir():
        candidates.append(pkg_templates)

    # Fallback: package directory itself
    if package_dir.is_dir():
        candidates.append(package_dir)

    return candidates if candidates else [Path(".")]


def _get_jinja_env() -> Environment:
    """Create and return a Jinja2 Environment configured for report templates.

    Returns:
        A Jinja2 Environment with FileSystemLoader pointing at the templates dir.
    """
    template_dirs = _find_template_dirs()
    loader = FileSystemLoader([str(d) for d in template_dirs])
    env = Environment(
        loader=loader,
        autoescape=select_autoescape([]),  # Markdown is not HTML — no autoescaping
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # Add custom filters
    env.filters["round1"] = lambda x: round(float(x), 1)
    env.filters["score_bar"] = _score_bar_text
    return env


def _score_bar_text(score: float, width: int = 20) -> str:
    """Convert a 0-100 score into a simple ASCII progress bar string.

    Args:
        score: Displacement score 0–100.
        width: Number of characters wide.

    Returns:
        ASCII bar string like ``[████████░░░░░░░░░░░░]``.
    """
    filled = int(round((score / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _risk_emoji(level: RiskLevel) -> str:
    """Return the emoji indicator for a risk level.

    Args:
        level: The RiskLevel.

    Returns:
        Emoji string.
    """
    return _RISK_EMOJIS.get(level, "⚪")


def _risk_color(level: RiskLevel) -> str:
    """Return the Rich color/style string for a risk level.

    Args:
        level: The RiskLevel.

    Returns:
        Rich markup style string.
    """
    return _RISK_COLORS.get(level, "white")


def _score_bar_rich(score: float, width: int = _SCORE_BAR_WIDTH) -> Text:
    """Build a Rich Text object representing a score bar with color.

    Args:
        score: Displacement score 0–100.
        width: Bar width in characters.

    Returns:
        Rich Text object.
    """
    filled = int(round((score / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled

    # Color based on score
    if score >= 75:
        color = "red"
    elif score >= 50:
        color = "yellow"
    elif score >= 25:
        color = "cyan"
    else:
        color = "green"

    text = Text()
    text.append("[", style="dim")
    text.append("█" * filled, style=color)
    text.append("░" * empty, style="dim")
    text.append("]", style="dim")
    return text


def _format_cost(cost: Optional[float]) -> str:
    """Format a monthly cost value for display.

    Args:
        cost: Monthly cost in USD, or None.

    Returns:
        Formatted string like '$1,234' or '—'.
    """
    if cost is None:
        return "—"
    if cost == 0.0:
        return "$0"
    return f"${cost:,.0f}"


def _format_team_size(size: Optional[int]) -> str:
    """Format a team size value for display.

    Args:
        size: Team size count, or None.

    Returns:
        Formatted string or '—'.
    """
    if size is None:
        return "—"
    return str(size)


# ---------------------------------------------------------------------------
# Reporter class
# ---------------------------------------------------------------------------


class Reporter:
    """Renders a ScanReport into terminal tables, Markdown, or JSON output.

    The Reporter is the primary output interface for the saas-risk-scan tool.
    It accepts a ScanReport and provides methods to render it in different formats.

    Attributes:
        report: The ScanReport to render.
        console: Rich Console for terminal output.
    """

    def __init__(
        self,
        report: ScanReport,
        console: Optional[Console] = None,
    ) -> None:
        """Initialize the Reporter with a ScanReport.

        Args:
            report: The ScanReport to render.
            console: Optional Rich Console (defaults to stderr-aware stdout console).
        """
        self.report = report
        self.console = console or Console()

    # -----------------------------------------------------------------------
    # Terminal table rendering
    # -----------------------------------------------------------------------

    def render_table(
        self,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
        show_dimensions: bool = False,
    ) -> None:
        """Render the scan report as a Rich terminal table.

        Displays a ranked table with tool names, categories, scores, risk levels,
        timelines, and top alternative suggestions. Optionally shows dimension
        breakdowns.

        Args:
            top_n: If set, show only the top N tools by displacement score.
            min_score: If set, filter to tools with score >= this value.
            show_dimensions: If True, add columns for each dimension score.
        """
        results = self._filter_results(top_n=top_n, min_score=min_score)

        if not results:
            self.console.print(
                "[yellow]No results to display (check --top / --min-score filters).[/yellow]"
            )
            return

        # ---- Header panel ----
        stats = self.report.summary_stats
        avg_score = stats.get("avg_displacement_score", 0.0)
        total_cost = stats.get("total_monthly_cost_usd")
        enriched = stats.get("tools_enriched_by_llm", 0)

        header_parts = [
            f"[bold]Tools analyzed:[/bold] {self.report.total_tools}",
            f"[bold]Avg displacement score:[/bold] {avg_score}",
        ]
        if total_cost is not None:
            header_parts.append(f"[bold]Total monthly spend:[/bold] {_format_cost(total_cost)}")
        if enriched:
            header_parts.append(f"[bold]LLM-enriched:[/bold] {enriched} tool(s)")
        if self.report.generated_at:
            header_parts.append(f"[bold]Generated:[/bold] {self.report.generated_at[:19]}")

        header_text = "   ·   ".join(header_parts)
        self.console.print()
        self.console.print(
            Panel(
                header_text,
                title="[bold blue]SaaS Displacement Risk Report[/bold blue]",
                border_style="blue",
            )
        )

        # ---- Risk summary counts ----
        risk_counts = stats.get("risk_level_counts", {})
        if isinstance(risk_counts, dict) and risk_counts:
            summary_parts: list[str] = []
            for level in (RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW):
                count = risk_counts.get(level.value, 0)
                if count:
                    color = _risk_color(level)
                    summary_parts.append(
                        f"[{color}]{_risk_emoji(level)} {count} {level.label}[/{color}]"
                    )
            if summary_parts:
                self.console.print("  " + "   ".join(summary_parts))
                self.console.print()

        # ---- Main table ----
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            show_lines=False,
            expand=False,
        )

        # Define columns
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Tool", style="bold white", min_width=14, max_width=25)
        table.add_column("Category", style="dim", min_width=12, max_width=20)
        table.add_column("Score", justify="right", min_width=6)
        table.add_column("Bar", min_width=_SCORE_BAR_WIDTH + 2)
        table.add_column("Risk", min_width=12)
        table.add_column("Timeline", min_width=11)
        table.add_column("Monthly Cost", justify="right", min_width=10)

        if show_dimensions:
            table.add_column("AutoR", justify="right", min_width=5)
            table.add_column("API", justify="right", min_width=5)
            table.add_column("WfCx", justify="right", min_width=5)
            table.add_column("DataS", justify="right", min_width=5)
            table.add_column("Inert", justify="right", min_width=5)

        table.add_column("Top Alternatives", min_width=20)
        table.add_column("Src", style="dim", min_width=3, max_width=4)

        # Add rows
        for result in results:
            level = result.risk_level
            color = _risk_color(level)
            score = result.displacement_score

            rank_str = str(result.rank) if result.rank is not None else "?"
            score_str = Text(f"{score:.1f}", style=color)
            risk_str = Text(
                f"{_risk_emoji(level)} {level.label}",
                style=color,
            )
            timeline_color = _TIMELINE_COLORS.get(result.timeline, "white")
            timeline_str = Text(result.timeline.display, style=timeline_color)
            bar = _score_bar_rich(score)
            cost_str = _format_cost(result.tool.monthly_cost_usd)
            alts_str = result.score.alternatives_display(max_items=_MAX_ALTS_IN_TABLE)
            category_label = result.tool.category.value.replace("_", "\u00a0")

            # Source indicator
            src_map = {"knowledge_base": "KB", "llm": "AI", "default": "~"}
            src_str = src_map.get(result.source, "?")

            row: list[object] = [
                rank_str,
                result.tool_name[:24],
                category_label[:19],
                score_str,
                bar,
                risk_str,
                timeline_str,
                cost_str,
            ]

            if show_dimensions:
                s = result.score
                row.extend([
                    f"{s.task_automation_ratio:.1f}",
                    f"{s.api_openness:.1f}",
                    f"{s.workflow_complexity:.1f}",
                    f"{s.data_sensitivity:.1f}",
                    f"{s.incumbent_inertia:.1f}",
                ])

            row.extend([alts_str, src_str])
            table.add_row(*[str(r) if not isinstance(r, (Text)) else r for r in row])  # type: ignore[arg-type]

        self.console.print(table)

        # ---- Footer ----
        if top_n is not None and len(results) < self.report.total_tools:
            self.console.print(
                f"  [dim]Showing top {len(results)} of {self.report.total_tools} tools.[/dim]"
            )
        if min_score is not None:
            self.console.print(
                f"  [dim]Filtered: score ≥ {min_score} · "
                f"{len(results)} of {self.report.total_tools} tools shown.[/dim]"
            )

        # Legend
        self.console.print()
        self.console.print(
            "  [dim]Src: KB=Knowledge Base  AI=LLM Enriched  ~=Heuristic Default[/dim]"
        )
        self.console.print()

    def render_detail(
        self,
        result: AnalysisResult,
    ) -> None:
        """Render a detailed panel for a single AnalysisResult.

        Shows all dimension scores, rationale, and alternatives in a
        formatted terminal panel.

        Args:
            result: The AnalysisResult to display in detail.
        """
        level = result.risk_level
        color = _risk_color(level)
        score = result.displacement_score

        lines: list[str] = []
        lines.append(
            f"[bold]Category:[/bold] {result.tool.category.value}   "
            f"[bold]Score:[/bold] [{color}]{score:.1f}/100[/{color}]   "
            f"[bold]Risk:[/bold] [{color}]{_risk_emoji(level)} {level.label}[/{color}]   "
            f"[bold]Timeline:[/bold] {result.timeline.display}"
        )

        if result.tool.monthly_cost_usd is not None or result.tool.team_size is not None:
            meta: list[str] = []
            if result.tool.monthly_cost_usd is not None:
                meta.append(f"Monthly cost: {_format_cost(result.tool.monthly_cost_usd)}")
            if result.tool.team_size is not None:
                meta.append(f"Team size: {result.tool.team_size}")
            lines.append("  ·  ".join(meta))

        lines.append("")
        lines.append("[bold]Dimension Scores:[/bold]")
        s = result.score
        dims = [
            ("Task Automation Ratio", s.task_automation_ratio, "30%"),
            ("API Openness", s.api_openness, "20%"),
            ("Workflow Complexity", s.workflow_complexity, "15%"),
            ("Data Sensitivity", s.data_sensitivity, "20%"),
            ("Incumbent Inertia", s.incumbent_inertia, "15%"),
        ]
        for name, val, weight in dims:
            bar = _score_bar_text(val * 10, width=10)  # scale 0-10 → bar
            lines.append(f"  {name:<25} {val:4.1f}/10  {bar}  (weight {weight})")

        if result.score.rationale:
            lines.append("")
            lines.append("[bold]Rationale:[/bold]")
            lines.append(result.score.rationale)

        if result.score.alternatives:
            lines.append("")
            lines.append("[bold]Agentic Alternatives:[/bold]")
            for alt in result.score.alternatives:
                lines.append(f"  • {alt}")

        if result.score.enriched_by_llm:
            lines.append("")
            lines.append("[dim italic]✨ Scored via LLM enrichment[/dim italic]")

        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]{result.tool_name}[/bold]",
                border_style=color,
                expand=False,
            )
        )

    # -----------------------------------------------------------------------
    # Markdown rendering
    # -----------------------------------------------------------------------

    def render_markdown(
        self,
        output_path: Optional[Union[str, Path]] = None,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> str:
        """Render the scan report as a Jinja2-templated Markdown string.

        Args:
            output_path: If provided, write the Markdown to this file path.
            top_n: If set, include only the top N tools in the report.
            min_score: If set, filter to tools with score >= this value.

        Returns:
            The rendered Markdown string.

        Raises:
            RuntimeError: If the Jinja2 template cannot be found or rendered.
        """
        results = self._filter_results(top_n=top_n, min_score=min_score)
        env = _get_jinja_env()

        try:
            template = env.get_template(_TEMPLATE_FILENAME)
        except TemplateNotFound:
            # Fall back to inline template
            template = env.from_string(_INLINE_MD_TEMPLATE)

        template_context = self._build_template_context(results)
        rendered = template.render(**template_context)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")

        return rendered

    def _build_template_context(
        self,
        results: list[AnalysisResult],
    ) -> dict[str, object]:
        """Build the Jinja2 template context dictionary from the report.

        Args:
            results: Filtered list of AnalysisResult items to include.

        Returns:
            Dictionary of template variables.
        """
        stats = self.report.summary_stats
        risk_counts = stats.get("risk_level_counts", {})

        return {
            "report": self.report,
            "results": results,
            "generated_at": self.report.generated_at[:19] if self.report.generated_at else "",
            "total_tools": self.report.total_tools,
            "shown_tools": len(results),
            "avg_score": stats.get("avg_displacement_score", 0.0),
            "total_monthly_cost": stats.get("total_monthly_cost_usd"),
            "enrichment_enabled": self.report.enrichment_enabled,
            "tools_enriched_by_llm": stats.get("tools_enriched_by_llm", 0),
            "risk_counts": {
                "critical": risk_counts.get("critical", 0),
                "high": risk_counts.get("high", 0),
                "medium": risk_counts.get("medium", 0),
                "low": risk_counts.get("low", 0),
            },
            # Helpers
            "RiskLevel": RiskLevel,
            "ReplacementTimeline": ReplacementTimeline,
            "risk_emoji": _risk_emoji,
            "format_cost": _format_cost,
            "score_bar": _score_bar_text,
        }

    # -----------------------------------------------------------------------
    # JSON rendering
    # -----------------------------------------------------------------------

    def render_json(
        self,
        output_path: Optional[Union[str, Path]] = None,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
        indent: int = 2,
    ) -> str:
        """Render the scan report as a JSON string.

        Args:
            output_path: If provided, write the JSON to this file path.
            top_n: If set, include only the top N tools.
            min_score: If set, filter to tools with score >= this value.
            indent: JSON indentation level (default 2).

        Returns:
            The rendered JSON string.
        """
        results = self._filter_results(top_n=top_n, min_score=min_score)

        # Build export dict from report, but replace results with filtered list
        export: dict[str, object] = {
            "generated_at": self.report.generated_at,
            "total_tools": self.report.total_tools,
            "shown_tools": len(results),
            "enrichment_enabled": self.report.enrichment_enabled,
            "summary_stats": self.report.summary_stats,
            "results": [r.to_dict() for r in results],
        }

        json_str = json.dumps(export, indent=indent, ensure_ascii=False, default=str)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json_str, encoding="utf-8")

        return json_str

    # -----------------------------------------------------------------------
    # Filtering helper
    # -----------------------------------------------------------------------

    def _filter_results(
        self,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[AnalysisResult]:
        """Filter and slice the report's results list.

        Results are already sorted by displacement score descending
        (as produced by ScanReport.rank_results()).

        Args:
            top_n: If set, return at most this many results.
            min_score: If set, return only results with score >= min_score.

        Returns:
            Filtered list of AnalysisResult items.
        """
        results = list(self.report.results)

        if min_score is not None:
            results = [r for r in results if r.displacement_score >= min_score]

        if top_n is not None:
            results = results[:top_n]

        return results


# ---------------------------------------------------------------------------
# Convenience render function
# ---------------------------------------------------------------------------


def render(
    report: ScanReport,
    fmt: str = "table",
    output_path: Optional[Union[str, Path]] = None,
    top_n: Optional[int] = None,
    min_score: Optional[float] = None,
    show_dimensions: bool = False,
    console: Optional[Console] = None,
) -> Optional[str]:
    """Convenience function to render a ScanReport in the specified format.

    Args:
        report: The ScanReport to render.
        fmt: Output format: 'table', 'markdown', or 'json'.
        output_path: For file-based formats, write output here.
        top_n: Show only the top N tools.
        min_score: Filter to tools with score >= this value.
        show_dimensions: For table format, show dimension score columns.
        console: Optional Rich Console for table output.

    Returns:
        The rendered string for 'markdown' and 'json' formats.
        None for 'table' (output goes directly to the console).

    Raises:
        ValueError: If fmt is not a supported format string.
    """
    fmt = fmt.lower().strip()
    reporter = Reporter(report=report, console=console)

    if fmt == "table":
        reporter.render_table(
            top_n=top_n,
            min_score=min_score,
            show_dimensions=show_dimensions,
        )
        return None
    elif fmt == "markdown":
        return reporter.render_markdown(
            output_path=output_path,
            top_n=top_n,
            min_score=min_score,
        )
    elif fmt == "json":
        return reporter.render_json(
            output_path=output_path,
            top_n=top_n,
            min_score=min_score,
        )
    else:
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported formats: table, markdown, json"
        )


def load_report_from_json(json_path: Union[str, Path]) -> ScanReport:
    """Load a ScanReport from a previously exported JSON file.

    This function deserializes a JSON file produced by Reporter.render_json()
    back into a ScanReport object, enabling the 'export' CLI command to
    convert saved results to other formats.

    Args:
        json_path: Path to the JSON report file.

    Returns:
        A reconstructed ScanReport (results will be pre-ranked).

    Raises:
        ValueError: If the JSON file cannot be parsed or has invalid structure.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON report file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON report: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Expected JSON report to be a JSON object (dict)")

    # Reconstruct AnalysisResult objects from the serialized dicts
    from saas_risk_scan.models import (
        AnalysisResult,
        RiskLevel,
        RiskScore,
        SaasTool,
        ToolCategory,
    )

    results: list[AnalysisResult] = []
    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("Expected 'results' key to be a list in JSON report")

    for idx, raw in enumerate(raw_results):
        try:
            tool_data = raw["tool"]
            score_data = raw["score"]
            dims = score_data["dimensions"]

            tool = SaasTool(
                name=tool_data["name"],
                category=ToolCategory(tool_data["category"]),
                monthly_cost_usd=tool_data.get("monthly_cost_usd"),
                team_size=tool_data.get("team_size"),
                notes=tool_data.get("notes"),
            )
            risk_score = RiskScore(
                task_automation_ratio=dims["task_automation_ratio"],
                api_openness=dims["api_openness"],
                workflow_complexity=dims["workflow_complexity"],
                data_sensitivity=dims["data_sensitivity"],
                incumbent_inertia=dims["incumbent_inertia"],
                displacement_score=score_data["displacement_score"],
                risk_level=RiskLevel(score_data["risk_level"]),
                timeline=ReplacementTimeline(score_data["timeline"]),
                alternatives=score_data.get("alternatives", []),
                rationale=score_data.get("rationale"),
                enriched_by_llm=score_data.get("enriched_by_llm", False),
            )
            result = AnalysisResult(
                tool=tool,
                score=risk_score,
                rank=raw.get("rank"),
                source=raw.get("source", "default"),
            )
            results.append(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to reconstruct result #{idx + 1} from JSON: {exc}"
            ) from exc

    report = ScanReport(
        results=results,
        generated_at=data.get("generated_at", ""),
        total_tools=data.get("total_tools", len(results)),
        enrichment_enabled=data.get("enrichment_enabled", False),
        summary_stats=data.get("summary_stats", {}),
    )
    return report


# ---------------------------------------------------------------------------
# Inline Markdown template fallback
# Used when templates/report.md.j2 cannot be found on disk.
# ---------------------------------------------------------------------------

_INLINE_MD_TEMPLATE = """\
# SaaS Displacement Risk Report

Generated: {{ generated_at }}  ·  Tools analyzed: {{ total_tools }}{% if shown_tools != total_tools %}  ·  Showing: {{ shown_tools }}{% endif %}  ·  Avg score: {{ avg_score | round1 }}
{% if total_monthly_cost is not none %}
**Total monthly SaaS spend:** ${{ "{:,.0f}".format(total_monthly_cost) }}
{% endif %}
{% if enrichment_enabled %}
> ✨ LLM enrichment was active for this scan ({{ tools_enriched_by_llm }} tool(s) enriched).
{% endif %}
---

## Executive Summary

{% set critical = risk_counts.critical %}
{% set high = risk_counts.high %}
{% set medium = risk_counts.medium %}
{% set low = risk_counts.low %}
{% if critical > 0 %}
**{{ critical }}** tool(s) at 🔴 **Critical** risk of near-term AI displacement (**< 12 months**).
{% endif %}
{% if high > 0 %}
**{{ high }}** tool(s) face 🟠 **High** displacement risk within **12–24 months**.
{% endif %}
{% if medium > 0 %}
**{{ medium }}** tool(s) face 🟡 **Medium** risk over a **24–36 month** horizon.
{% endif %}
{% if low > 0 %}
**{{ low }}** tool(s) have 🟢 **Low** displacement risk (**36+ months**).
{% endif %}

---

## Risk Rankings

| # | Tool | Category | Score | Risk | Timeline | Monthly Cost | Top Alternatives |
|---|------|----------|------:|------|----------|-------------:|------------------|
{% for r in results %}
| {{ r.rank }} | **{{ r.tool_name }}** | {{ r.tool.category.value }} | {{ r.score.displacement_score | round1 }} | {{ risk_emoji(r.risk_level) }} {{ r.risk_level.label }} | {{ r.timeline.display }} | {{ format_cost(r.tool.monthly_cost_usd) }} | {{ r.score.alternatives_display(3) }} |
{% endfor %}

---

## Detailed Analysis

{% for r in results %}
### {{ r.rank }}. {{ r.tool_name }}

{{ score_bar(r.score.displacement_score) }} **{{ r.score.displacement_score | round1 }}/100** · {{ risk_emoji(r.risk_level) }} {{ r.risk_level.label }} · {{ r.timeline.display }}

**Category:** {{ r.tool.category.value }}
{% if r.tool.monthly_cost_usd is not none %}**Monthly Cost:** {{ format_cost(r.tool.monthly_cost_usd) }}  {% endif %}{% if r.tool.team_size is not none %}**Team Size:** {{ r.tool.team_size }}{% endif %}

**Dimension Scores:**

| Dimension | Score | Weight |
|-----------|------:|-------|
| Task Automation Ratio | {{ r.score.task_automation_ratio }}/10 | 30% |
| API Openness | {{ r.score.api_openness }}/10 | 20% |
| Workflow Complexity | {{ r.score.workflow_complexity }}/10 | 15% |
| Data Sensitivity | {{ r.score.data_sensitivity }}/10 | 20% |
| Incumbent Inertia | {{ r.score.incumbent_inertia }}/10 | 15% |

{% if r.score.rationale %}
> {{ r.score.rationale }}
{% endif %}
{% if r.score.alternatives %}
**Agentic Alternatives:**
{% for alt in r.score.alternatives %}
- {{ alt }}
{% endfor %}
{% endif %}
{% if r.score.enriched_by_llm %}
*✨ Scored via LLM enrichment*
{% endif %}
{% if r.source == 'knowledge_base' %}
*Source: Knowledge Base*
{% elif r.source == 'llm' %}
*Source: LLM Enrichment*
{% else %}
*Source: Heuristic Defaults*
{% endif %}

---
{% endfor %}

## Scoring Methodology

Each tool is scored on five dimensions (0–10), combined into a weighted **Displacement Score** (0–100):

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Task Automation Ratio | 30% | How much of the tool's core value is pure task execution vs. judgment |
| API Openness | 20% | Quality and completeness of public APIs / webhook support |
| Workflow Complexity | 15% | Depth of embedding in multi-step human workflows (lower = higher risk) |
| Data Sensitivity | 20% | Risk and friction of data migration / lock-in (higher = lower risk) |
| Incumbent Inertia | 15% | Organizational switching cost (higher = lower risk) |

### Score → Risk Mapping

| Score | Risk | Timeline |
|-------|------|----------|
| 75–100 | 🔴 Critical | < 12 months |
| 50–74 | 🟠 High | 12–24 months |
| 25–49 | 🟡 Medium | 24–36 months |
| 0–24 | 🟢 Low | 36+ months |

---

*Generated by [saas-risk-scan](https://github.com/example/saas-risk-scan)*
"""
