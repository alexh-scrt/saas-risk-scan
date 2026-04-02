"""Typer CLI entry point for the SaaS Risk Scan tool.

This module wires up the three CLI commands:
    - scan        : Load a YAML/JSON/CSV file and produce a risk report.
    - interactive : Collect tools via interactive prompts, then produce a report.
    - export      : Convert a previously saved JSON report to another format.

All commands share common options for output format, filtering, and enrichment.
Rich is used for all terminal output; progress indication is shown during LLM
enrichment via a Rich progress bar.

Environment variables:
    OPENAI_API_KEY          : Enable LLM enrichment (optional).
    SAAS_RISK_OPENAI_MODEL  : Override the default OpenAI model (optional).
    SAAS_RISK_DEFAULT_FORMAT: Override default output format (optional).
"""

from __future__ import annotations

import datetime
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.panel import Panel
from rich.text import Text

from saas_risk_scan import __version__
from saas_risk_scan.enricher import (
    enrich_stack,
    get_enrichment_summary,
    is_enrichment_available,
)
from saas_risk_scan.loader import (
    LoadError,
    load_file,
    load_interactive,
    supported_categories,
)
from saas_risk_scan.models import (
    SaasStack,
    ScanReport,
    ToolCategory,
)
from saas_risk_scan.reporter import Reporter, load_report_from_json, render
from saas_risk_scan.scorer import build_report

# ---------------------------------------------------------------------------
# Typer application setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="saas-risk-scan",
    help=(
        "SaaS Replacement Risk Analyzer — audit your SaaS stack for AI displacement risk.\n\n"
        "Score each tool across 5 dimensions and generate a ranked report with "
        "replacement timelines and agentic alternative suggestions."
    ),
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

# Shared Rich console for consistent terminal output
console = Console(stderr=False)
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Output format enum
# ---------------------------------------------------------------------------


class OutputFormat(str, Enum):  # noqa: N801
    """Supported output formats for CLI commands."""

    TABLE = "table"
    MARKDOWN = "markdown"
    JSON = "json"


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print the version string and exit."""
    if value:
        console.print(f"saas-risk-scan version [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """SaaS Replacement Risk Analyzer."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_timestamp() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _abort(message: str, exit_code: int = 1) -> None:
    """Print an error message to stderr and exit.

    Args:
        message: Error message to display.
        exit_code: Exit code (default 1).
    """
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=exit_code)


def _warn(message: str) -> None:
    """Print a warning message to stderr.

    Args:
        message: Warning message to display.
    """
    err_console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def _info(message: str) -> None:
    """Print an info message to stderr.

    Args:
        message: Info message to display.
    """
    err_console.print(f"[dim]{message}[/dim]")


def _run_enrichment(
    stack: SaasStack,
    enrich: bool,
) -> list:
    """Run enrichment (LLM or rule-based) on a SaasStack.

    Shows a progress indicator when LLM enrichment is active. Falls back
    gracefully to rule-based scoring if the API key is not set.

    Args:
        stack: The SaasStack to enrich.
        enrich: Whether LLM enrichment was requested.

    Returns:
        List of AnalysisResult items.
    """
    from saas_risk_scan.scorer import score_stack

    if not enrich:
        return score_stack(stack)

    if not is_enrichment_available():
        _warn(
            "--enrich was requested but OPENAI_API_KEY is not set. "
            "Falling back to rule-based scoring for all tools."
        )
        return score_stack(stack)

    # LLM enrichment with Rich progress display
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    results_list: list = []
    total = len(stack.tools)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Enriching tools via LLM...", total=total)

        def _progress_callback(tool_name: str, idx: int, _total: int) -> None:
            progress.update(
                task_id,
                advance=1,
                description=f"Enriching: {tool_name[:30]}...",
            )

        results_list = enrich_stack(
            stack,
            enrich_unknown_only=True,
            progress_callback=_progress_callback,
        )

    # Show enrichment summary
    summary = get_enrichment_summary(results_list)
    err_console.print(
        f"[dim]Enrichment complete: "
        f"{summary['knowledge_base']} KB · "
        f"{summary['llm']} LLM · "
        f"{summary['default']} heuristic[/dim]"
    )

    return results_list


def _output_report(
    report: ScanReport,
    fmt: OutputFormat,
    output: Optional[Path],
    top_n: Optional[int],
    min_score: Optional[float],
    show_dimensions: bool = False,
) -> None:
    """Render and output a ScanReport in the specified format.

    For table format, writes to the console. For markdown/json, either writes
    to a file (if --output is specified) or prints to stdout.

    Args:
        report: The ScanReport to render.
        fmt: The output format.
        output: Optional output file path.
        top_n: Show only top N tools.
        min_score: Filter to tools with score >= this value.
        show_dimensions: Show dimension score columns in table format.
    """
    fmt_str = fmt.value

    if fmt_str == "table":
        reporter = Reporter(report=report, console=console)
        reporter.render_table(
            top_n=top_n,
            min_score=min_score,
            show_dimensions=show_dimensions,
        )
        # If output path requested for table, also write markdown
        if output is not None:
            _warn(
                f"Table format cannot be saved to a file; use --format markdown or json. "
                f"Ignoring --output {output}."
            )
        return

    # Non-table formats: get rendered string
    rendered: Optional[str] = render(
        report=report,
        fmt=fmt_str,
        output_path=output,
        top_n=top_n,
        min_score=min_score,
    )

    # If no output file, print to stdout
    if output is None and rendered is not None:
        print(rendered)  # noqa: T201 — intentional stdout output
    elif output is not None:
        _info(f"Report written to: {output}")


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


@app.command("scan")
def scan_command(
    input_file: Path = typer.Argument(
        ...,
        help="Path to YAML, JSON, or CSV file listing your SaaS tools.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="INPUT_FILE",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If omitted, prints to stdout (or terminal for table).",
        metavar="FILE",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table (Rich terminal), markdown, or json.",
        case_sensitive=False,
    ),
    enrich: bool = typer.Option(
        False,
        "--enrich",
        "-e",
        help=(
            "Enable LLM enrichment for unknown tools via OpenAI. "
            "Requires OPENAI_API_KEY environment variable."
        ),
        is_flag=True,
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top",
        "-n",
        help="Show only the top N tools by displacement score.",
        min=1,
        metavar="N",
    ),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        help="Filter: only show tools with displacement score >= this value (0–100).",
        min=0.0,
        max=100.0,
        metavar="SCORE",
    ),
    show_dimensions: bool = typer.Option(
        False,
        "--dimensions",
        "-d",
        help="Show individual dimension score columns in table output.",
        is_flag=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional diagnostic information during scanning.",
        is_flag=True,
    ),
) -> None:
    """Scan a SaaS tool list from a file and produce a displacement risk report.

    Reads tool definitions from INPUT_FILE (YAML, JSON, or CSV) and scores each
    tool across five risk dimensions. Outputs a ranked report showing displacement
    risk, estimated timelines, and agentic alternative suggestions.

    Examples:

    \b
        # Display a Rich terminal table
        saas-risk-scan scan examples/sample_stack.yaml

    \b
        # Export a Markdown report
        saas-risk-scan scan my_stack.yaml --format markdown --output report.md

    \b
        # Export JSON with LLM enrichment for unknown tools
        saas-risk-scan scan my_stack.yaml --enrich --format json --output results.json

    \b
        # Show only the top 5 highest-risk tools
        saas-risk-scan scan my_stack.yaml --top 5
    """
    # ---- Load the input file ----
    if verbose:
        _info(f"Loading tools from: {input_file}")

    try:
        stack = load_file(input_file)
    except LoadError as exc:
        _abort(f"Failed to load input file: {exc}")
        return  # unreachable, satisfies type checker

    if verbose:
        _info(f"Loaded {stack.tool_count()} tool(s) from {input_file.name}")

    # Warn about duplicate tool names
    dupes = stack.duplicate_names()
    if dupes:
        _warn(
            f"Duplicate tool names detected (case-insensitive): "
            f"{', '.join(dupes)}. They will each be scored independently."
        )

    # ---- Score / enrich ----
    pre_scored = _run_enrichment(stack, enrich=enrich)

    # ---- Build report ----
    timestamp = _get_timestamp()
    report = build_report(
        stack=stack,
        generated_at=timestamp,
        enrichment_enabled=enrich and is_enrichment_available(),
        pre_scored_results=pre_scored if enrich else None,
    )

    if verbose:
        stats = report.summary_stats
        _info(
            f"Scoring complete. Avg score: {stats.get('avg_displacement_score', '?')} · "
            f"Tools: {report.total_tools}"
        )

    # ---- Output ----
    _output_report(
        report=report,
        fmt=fmt,
        output=output,
        top_n=top_n,
        min_score=min_score,
        show_dimensions=show_dimensions,
    )


# ---------------------------------------------------------------------------
# interactive command
# ---------------------------------------------------------------------------


@app.command("interactive")
def interactive_command(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If omitted, prints to stdout (or terminal for table).",
        metavar="FILE",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table (Rich terminal), markdown, or json.",
        case_sensitive=False,
    ),
    enrich: bool = typer.Option(
        False,
        "--enrich",
        "-e",
        help=(
            "Enable LLM enrichment for unknown tools via OpenAI. "
            "Requires OPENAI_API_KEY environment variable."
        ),
        is_flag=True,
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top",
        "-n",
        help="Show only the top N tools by displacement score.",
        min=1,
        metavar="N",
    ),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        help="Filter: only show tools with displacement score >= this value (0–100).",
        min=0.0,
        max=100.0,
        metavar="SCORE",
    ),
    show_dimensions: bool = typer.Option(
        False,
        "--dimensions",
        "-d",
        help="Show individual dimension score columns in table output.",
        is_flag=True,
    ),
) -> None:
    """Enter SaaS tools interactively via prompts and generate a risk report.

    Guides you through entering each tool's name, category, and optional
    metadata (cost, team size, notes) one at a time. When you're done adding
    tools, the report is generated and displayed.

    Examples:

    \b
        # Interactive session → Rich terminal table
        saas-risk-scan interactive

    \b
        # Interactive session → save Markdown report
        saas-risk-scan interactive --format markdown --output report.md

    \b
        # Interactive session with LLM enrichment
        saas-risk-scan interactive --enrich
    """
    console.print()
    console.print(
        Panel(
            "[bold]SaaS Risk Scan — Interactive Mode[/bold]\n\n"
            "Enter your SaaS tools one at a time. "
            "Press [bold cyan]Enter[/bold cyan] with a blank name when done.\n"
            "Required fields: [bold]name[/bold], [bold]category[/bold]  ·  "
            "Optional: cost, team size, notes",
            border_style="cyan",
        )
    )

    # Show available categories
    categories = supported_categories()
    console.print(
        "\n[bold]Available categories:[/bold] "
        + ", ".join(f"[cyan]{c}[/cyan]" for c in categories)
        + "\n"
    )

    tools_data: list[dict[str, object]] = []
    tool_number = 1

    while True:
        console.print(f"[bold]Tool #{tool_number}[/bold]")

        # Tool name
        name = Prompt.ask(
            "  Name [dim](blank to finish)[/dim]",
            default="",
            console=console,
        ).strip()

        if not name:
            if not tools_data:
                _warn("No tools entered. Please add at least one tool.")
                continue
            break

        # Category
        while True:
            category_input = Prompt.ask(
                "  Category",
                default="other",
                console=console,
            ).strip().lower()

            if category_input in categories:
                break

            # Try prefix matching
            matches = [c for c in categories if c.startswith(category_input)]
            if len(matches) == 1:
                category_input = matches[0]
                console.print(f"  [dim]→ Using: {category_input}[/dim]")
                break
            elif len(matches) > 1:
                console.print(
                    f"  [yellow]Ambiguous category '{category_input}'. "
                    f"Matches: {', '.join(matches)}. Please be more specific.[/yellow]"
                )
            else:
                console.print(
                    f"  [yellow]Unknown category '{category_input}'. "
                    f"Valid: {', '.join(categories)}[/yellow]"
                )

        # Optional fields
        cost_str = Prompt.ask(
            "  Monthly cost (USD) [dim](optional, blank to skip)[/dim]",
            default="",
            console=console,
        ).strip()
        monthly_cost: Optional[float] = None
        if cost_str:
            try:
                monthly_cost = float(cost_str.replace("$", "").replace(",", ""))
            except ValueError:
                _warn(f"Could not parse cost '{cost_str}'; skipping.")

        team_str = Prompt.ask(
            "  Team size (users) [dim](optional, blank to skip)[/dim]",
            default="",
            console=console,
        ).strip()
        team_size: Optional[int] = None
        if team_str:
            try:
                team_size = int(float(team_str))
            except ValueError:
                _warn(f"Could not parse team size '{team_str}'; skipping.")

        notes = Prompt.ask(
            "  Notes [dim](optional context, blank to skip)[/dim]",
            default="",
            console=console,
        ).strip() or None

        raw: dict[str, object] = {"name": name, "category": category_input}
        if monthly_cost is not None:
            raw["monthly_cost_usd"] = monthly_cost
        if team_size is not None:
            raw["team_size"] = team_size
        if notes:
            raw["notes"] = notes

        tools_data.append(raw)
        console.print(f"  [green]✓ Added {name}[/green]\n")
        tool_number += 1

        # Ask if user wants to add more
        add_more = Confirm.ask(
            "  Add another tool?",
            default=True,
            console=console,
        )
        if not add_more:
            break
        console.print()

    # ---- Build the stack ----
    console.print()
    _info(f"Processing {len(tools_data)} tool(s)...")

    try:
        stack = load_interactive(tools_data)
    except LoadError as exc:
        _abort(f"Failed to build tool stack: {exc}")
        return

    # ---- Score / enrich ----
    pre_scored = _run_enrichment(stack, enrich=enrich)

    # ---- Build report ----
    timestamp = _get_timestamp()
    report = build_report(
        stack=stack,
        generated_at=timestamp,
        enrichment_enabled=enrich and is_enrichment_available(),
        pre_scored_results=pre_scored if enrich else None,
    )

    # ---- Output ----
    _output_report(
        report=report,
        fmt=fmt,
        output=output,
        top_n=top_n,
        min_score=min_score,
        show_dimensions=show_dimensions,
    )


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


@app.command("export")
def export_command(
    input_json: Path = typer.Argument(
        ...,
        help="Path to a previously saved JSON results file (from 'scan --format json').",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="INPUT_JSON",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If omitted, prints to stdout.",
        metavar="FILE",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.MARKDOWN,
        "--format",
        "-f",
        help="Output format: markdown or json. (table is not supported for export)",
        case_sensitive=False,
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top",
        "-n",
        help="Show only the top N tools by displacement score.",
        min=1,
        metavar="N",
    ),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        help="Filter: only show tools with displacement score >= this value (0–100).",
        min=0.0,
        max=100.0,
        metavar="SCORE",
    ),
) -> None:
    """Convert a saved JSON results file to another format.

    Loads a JSON file previously generated by 'scan --format json' and
    re-renders it as Markdown or JSON with optional filtering.

    Examples:

    \b
        # Convert saved JSON results to Markdown
        saas-risk-scan export results.json --output report.md

    \b
        # Print Markdown to stdout
        saas-risk-scan export results.json --format markdown

    \b
        # Re-export JSON with only top 10 tools
        saas-risk-scan export results.json --format json --top 10 --output top10.json
    """
    # Table format is not meaningful for export (no live console context)
    if fmt == OutputFormat.TABLE:
        _warn(
            "Table format is not supported for 'export'. "
            "Using 'markdown' instead. Pass --format json for JSON output."
        )
        fmt = OutputFormat.MARKDOWN

    # ---- Load the JSON report ----
    try:
        report = load_report_from_json(input_json)
    except FileNotFoundError as exc:
        _abort(str(exc))
        return
    except ValueError as exc:
        _abort(f"Failed to load JSON report: {exc}")
        return

    _info(
        f"Loaded report from {input_json.name}: "
        f"{report.total_tools} tool(s), generated {report.generated_at[:19]}"
    )

    # ---- Output ----
    _output_report(
        report=report,
        fmt=fmt,
        output=output,
        top_n=top_n,
        min_score=min_score,
    )


# ---------------------------------------------------------------------------
# info command (bonus: show KB stats and model weights)
# ---------------------------------------------------------------------------


@app.command("info")
def info_command() -> None:
    """Show information about the knowledge base, scoring model, and configuration.

    Displays the number of known tools, scoring dimension weights, current
    configuration (API key status, model), and supported categories.

    Example:

    \b
        saas-risk-scan info
    """
    from saas_risk_scan.knowledge_base import knowledge_base_size, known_tool_names
    from saas_risk_scan.scorer import get_dimension_weights

    console.print()
    console.print(
        Panel(
            f"[bold cyan]saas-risk-scan[/bold cyan] v{__version__}\n"
            "SaaS Replacement Risk Analyzer — AI Displacement Audit Tool",
            border_style="cyan",
        )
    )

    # Configuration
    console.print("\n[bold]Configuration[/bold]")
    api_key_set = is_enrichment_available()
    api_status = (
        "[green]✓ Set[/green]" if api_key_set else "[dim]Not set (LLM enrichment disabled)[/dim]"
    )
    console.print(f"  OPENAI_API_KEY     : {api_status}")

    import os
    model = os.environ.get("SAAS_RISK_OPENAI_MODEL", "gpt-4o-mini (default)")
    console.print(f"  OpenAI model       : [cyan]{model}[/cyan]")
    default_fmt = os.environ.get("SAAS_RISK_DEFAULT_FORMAT", "table (default)")
    console.print(f"  Default format     : [cyan]{default_fmt}[/cyan]")

    # Knowledge base stats
    console.print("\n[bold]Knowledge Base[/bold]")
    kb_size = knowledge_base_size()
    console.print(f"  Tools indexed      : [bold cyan]{kb_size}[/bold cyan]")
    names = known_tool_names()
    console.print(
        f"  Known tools        : {', '.join(names[:10])}"
        + (f" ... and {len(names) - 10} more" if len(names) > 10 else "")
    )

    # Scoring model weights
    console.print("\n[bold]Scoring Model — Dimension Weights[/bold]")
    weights = get_dimension_weights()
    weight_rows = [
        ("Task Automation Ratio", weights["task_automation_ratio"], "↑ Higher = more displaceable"),
        ("API Openness", weights["api_openness"], "↑ Higher = more displaceable"),
        ("Workflow Complexity", weights["workflow_complexity"], "↓ Lower = more displaceable"),
        ("Data Sensitivity", weights["data_sensitivity"], "↓ Higher = less displaceable"),
        ("Incumbent Inertia", weights["incumbent_inertia"], "↓ Higher = less displaceable"),
    ]
    for name, w, note in weight_rows:
        console.print(f"  {name:<28} [bold]{w:.0f}%[/bold]  [dim]{note}[/dim]")

    # Risk level bands
    console.print("\n[bold]Risk Level Bands[/bold]")
    bands = [
        ("🔴 Critical", "75–100", "< 12 months"),
        ("🟠 High", "50–74", "12–24 months"),
        ("🟡 Medium", "25–49", "24–36 months"),
        ("🟢 Low", "0–24", "36+ months"),
    ]
    for label, score_range, timeline in bands:
        console.print(f"  {label:<14} Score {score_range:<8} Timeline: {timeline}")

    # Supported categories
    console.print("\n[bold]Supported Categories[/bold]")
    cats = supported_categories()
    console.print("  " + ", ".join(f"[cyan]{c}[/cyan]" for c in cats))
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
