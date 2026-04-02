"""Loader module for parsing SaaS tool lists from YAML, JSON, and CSV input files.

This module provides functions to load and validate SaaS tool definitions from
disk or string content. It normalizes input across three supported formats into
validated SaasTool and SaasStack Pydantic models.

Supported formats:
    - YAML (.yaml, .yml): expected to have a top-level ``tools`` key
    - JSON (.json): expected to have a top-level ``tools`` key
    - CSV (.csv): header row with columns matching SaasTool fields

All unknown fields are silently ignored; missing required fields raise
a descriptive LoadError with the offending row number where applicable.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from saas_risk_scan.models import SaasStack, SaasTool, ToolCategory


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class LoadError(Exception):
    """Raised when input file loading or validation fails.

    Attributes:
        message: Human-readable error description.
        source: Optional file path or format label for context.
        row: Optional 1-based row number for CSV errors.
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        row: Optional[int] = None,
    ) -> None:
        """Initialize LoadError with message and optional context."""
        self.message = message
        self.source = source
        self.row = row
        parts = []
        if source:
            parts.append(f"[{source}]")
        if row is not None:
            parts.append(f"row {row}")
        parts.append(message)
        super().__init__(" ".join(parts))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml", ".json", ".csv"})


def detect_format(path: Path) -> str:
    """Detect the file format from its extension.

    Args:
        path: Path to the input file.

    Returns:
        One of ``'yaml'``, ``'json'``, or ``'csv'``.

    Raises:
        LoadError: If the extension is not supported.
    """
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        return "yaml"
    elif ext == ".json":
        return "json"
    elif ext == ".csv":
        return "csv"
    else:
        raise LoadError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            source=str(path),
        )


# ---------------------------------------------------------------------------
# Low-level row → SaasTool coercion
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: frozenset[str] = frozenset({"name", "category"})
_OPTIONAL_FIELDS: frozenset[str] = frozenset({"monthly_cost_usd", "team_size", "notes"})
_ALL_KNOWN_FIELDS: frozenset[str] = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


def _coerce_tool_dict(
    raw: dict[str, object],
    source: str = "input",
    row: Optional[int] = None,
) -> SaasTool:
    """Coerce a raw dictionary into a validated SaasTool model.

    Handles type coercion for optional numeric fields and strips unknown keys.

    Args:
        raw: Raw dictionary with tool field data.
        source: Label for error messages (e.g., file name or format).
        row: Optional 1-based row index for CSV error context.

    Returns:
        A validated SaasTool instance.

    Raises:
        LoadError: If required fields are missing or validation fails.
    """
    # Check for required fields
    missing = _REQUIRED_FIELDS - set(str(k) for k in raw.keys())
    if missing:
        raise LoadError(
            f"Missing required fields: {', '.join(sorted(missing))}",
            source=source,
            row=row,
        )

    # Build a clean dict with only known fields
    clean: dict[str, object] = {}
    for field in _ALL_KNOWN_FIELDS:
        if field in raw:
            val = raw[field]
            # Coerce empty strings to None for optional fields
            if field in _OPTIONAL_FIELDS and isinstance(val, str) and val.strip() == "":
                val = None
            clean[field] = val

    # Type coercion for numeric optional fields from strings (CSV case)
    if "monthly_cost_usd" in clean and clean["monthly_cost_usd"] is not None:
        try:
            clean["monthly_cost_usd"] = float(str(clean["monthly_cost_usd"]))
        except (ValueError, TypeError) as exc:
            raise LoadError(
                f"Invalid value for 'monthly_cost_usd': {clean['monthly_cost_usd']!r} "
                f"(expected a number)",
                source=source,
                row=row,
            ) from exc

    if "team_size" in clean and clean["team_size"] is not None:
        try:
            clean["team_size"] = int(float(str(clean["team_size"])))
        except (ValueError, TypeError) as exc:
            raise LoadError(
                f"Invalid value for 'team_size': {clean['team_size']!r} "
                f"(expected an integer)",
                source=source,
                row=row,
            ) from exc

    # Attempt Pydantic validation
    try:
        return SaasTool(**clean)  # type: ignore[arg-type]
    except ValidationError as exc:
        # Extract concise error messages from Pydantic
        errors = exc.errors(include_url=False)
        messages = [
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in errors
        ]
        raise LoadError(
            f"Validation error: {'; '.join(messages)}",
            source=source,
            row=row,
        ) from exc


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------


def _parse_yaml_content(content: str, source: str = "yaml") -> list[SaasTool]:
    """Parse YAML string content into a list of SaasTool instances.

    Expected structure::

        tools:
          - name: Zapier
            category: automation
            monthly_cost_usd: 599

    Also accepts a bare list at the top level::

        - name: Zapier
          category: automation

    Args:
        content: Raw YAML string.
        source: Label for error messages.

    Returns:
        List of validated SaasTool instances.

    Raises:
        LoadError: On YAML parse errors or validation failures.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise LoadError(f"YAML parse error: {exc}", source=source) from exc

    if data is None:
        raise LoadError("YAML file is empty or contains only null", source=source)

    # Accept either {tools: [...]} or bare list
    if isinstance(data, dict):
        if "tools" not in data:
            raise LoadError(
                "YAML file must have a top-level 'tools' key containing a list of tools. "
                "Example: 'tools:\n  - name: Zapier\n    category: automation'",
                source=source,
            )
        raw_tools = data["tools"]
    elif isinstance(data, list):
        raw_tools = data
    else:
        raise LoadError(
            f"Expected YAML to contain a mapping with 'tools' key or a list, "
            f"got {type(data).__name__}",
            source=source,
        )

    if not isinstance(raw_tools, list):
        raise LoadError(
            f"Expected 'tools' to be a list, got {type(raw_tools).__name__}",
            source=source,
        )

    tools: list[SaasTool] = []
    for idx, raw in enumerate(raw_tools, start=1):
        if not isinstance(raw, dict):
            raise LoadError(
                f"Each tool entry must be a mapping (dict), got {type(raw).__name__}",
                source=source,
                row=idx,
            )
        tools.append(_coerce_tool_dict(raw, source=source, row=idx))

    return tools


def _parse_json_content(content: str, source: str = "json") -> list[SaasTool]:
    """Parse JSON string content into a list of SaasTool instances.

    Expected structure::

        {"tools": [{"name": "Zapier", "category": "automation"}]}

    Also accepts a bare JSON array::

        [{"name": "Zapier", "category": "automation"}]

    Args:
        content: Raw JSON string.
        source: Label for error messages.

    Returns:
        List of validated SaasTool instances.

    Raises:
        LoadError: On JSON parse errors or validation failures.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LoadError(f"JSON parse error: {exc}", source=source) from exc

    if isinstance(data, dict):
        if "tools" not in data:
            raise LoadError(
                "JSON file must have a top-level 'tools' key. "
                "Example: '{\"tools\": [{\"name\": \"Zapier\", \"category\": \"automation\"}]}'",
                source=source,
            )
        raw_tools = data["tools"]
    elif isinstance(data, list):
        raw_tools = data
    else:
        raise LoadError(
            f"Expected JSON to contain an object with 'tools' key or an array, "
            f"got {type(data).__name__}",
            source=source,
        )

    if not isinstance(raw_tools, list):
        raise LoadError(
            f"Expected 'tools' to be a list, got {type(raw_tools).__name__}",
            source=source,
        )

    tools: list[SaasTool] = []
    for idx, raw in enumerate(raw_tools, start=1):
        if not isinstance(raw, dict):
            raise LoadError(
                f"Each tool entry must be a JSON object, got {type(raw).__name__}",
                source=source,
                row=idx,
            )
        tools.append(_coerce_tool_dict(raw, source=source, row=idx))

    return tools


# CSV column name aliases: maps alternative column names to canonical field names
_CSV_COLUMN_ALIASES: dict[str, str] = {
    "tool_name": "name",
    "tool": "name",
    "cost": "monthly_cost_usd",
    "monthly_cost": "monthly_cost_usd",
    "cost_usd": "monthly_cost_usd",
    "users": "team_size",
    "seats": "team_size",
    "size": "team_size",
    "note": "notes",
    "description": "notes",
    "comment": "notes",
}


def _normalize_csv_header(header: str) -> str:
    """Normalize a CSV column header to a canonical field name.

    Args:
        header: Raw CSV column header string.

    Returns:
        Canonical field name (lowercased and alias-resolved).
    """
    normalized = header.strip().lower().replace(" ", "_").replace("-", "_")
    return _CSV_COLUMN_ALIASES.get(normalized, normalized)


def _parse_csv_content(content: str, source: str = "csv") -> list[SaasTool]:
    """Parse CSV string content into a list of SaasTool instances.

    Expected header columns (case-insensitive, aliases supported)::

        name, category, monthly_cost_usd, team_size, notes

    Args:
        content: Raw CSV string.
        source: Label for error messages.

    Returns:
        List of validated SaasTool instances.

    Raises:
        LoadError: On CSV parse errors, missing headers, or validation failures.
    """
    reader = csv.DictReader(io.StringIO(content))

    if reader.fieldnames is None:
        raise LoadError("CSV file appears to be empty (no header row found)", source=source)

    # Normalize column names
    normalized_fieldnames = [_normalize_csv_header(f) for f in reader.fieldnames]

    # Check required columns
    missing_cols = _REQUIRED_FIELDS - set(normalized_fieldnames)
    if missing_cols:
        raise LoadError(
            f"CSV is missing required columns: {', '.join(sorted(missing_cols))}. "
            f"Found columns: {', '.join(str(f) for f in reader.fieldnames)}",
            source=source,
        )

    tools: list[SaasTool] = []
    for row_idx, row in enumerate(reader, start=2):  # start=2 because row 1 is header
        # Re-key the row using normalized column names
        normalized_row: dict[str, object] = {}
        for original_key, value in row.items():
            if original_key is None:
                continue
            canonical = _normalize_csv_header(original_key)
            normalized_row[canonical] = value

        # Skip entirely blank rows
        str_values = [str(v).strip() for v in normalized_row.values() if v is not None]
        if all(v == "" for v in str_values):
            continue

        tools.append(_coerce_tool_dict(normalized_row, source=source, row=row_idx))

    return tools


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_file(path: Union[str, Path]) -> SaasStack:
    """Load a SaaS tool list from a YAML, JSON, or CSV file.

    The format is determined by the file extension. The returned SaasStack
    is fully validated.

    Args:
        path: Path to the input file. Accepts str or pathlib.Path.

    Returns:
        A validated SaasStack containing all tools from the file.

    Raises:
        LoadError: If the file cannot be read, parsed, or validated.
        FileNotFoundError: If the file does not exist (re-raised as LoadError).
    """
    file_path = Path(path)
    source = file_path.name

    if not file_path.exists():
        raise LoadError(f"File not found: {file_path}", source=source)

    if not file_path.is_file():
        raise LoadError(f"Path is not a file: {file_path}", source=source)

    fmt = detect_format(file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LoadError(f"Cannot read file: {exc}", source=source) from exc

    return load_string(content, fmt=fmt, source=source)


def load_string(
    content: str,
    fmt: str,
    source: str = "string",
) -> SaasStack:
    """Load a SaaS tool list from a string in the given format.

    This function is useful for testing and for processing content that
    was not loaded directly from a file.

    Args:
        content: Raw string content of the tool list.
        fmt: Format string: ``'yaml'``, ``'json'``, or ``'csv'``.
        source: Optional label for error messages.

    Returns:
        A validated SaasStack containing all tools from the string.

    Raises:
        LoadError: If parsing or validation fails.
        ValueError: If ``fmt`` is not one of the supported formats.
    """
    fmt = fmt.lower().strip()

    if fmt in ("yaml", "yml"):
        tools = _parse_yaml_content(content, source=source)
    elif fmt == "json":
        tools = _parse_json_content(content, source=source)
    elif fmt == "csv":
        tools = _parse_csv_content(content, source=source)
    else:
        raise LoadError(
            f"Unknown format '{fmt}'. Supported formats: yaml, json, csv",
            source=source,
        )

    if not tools:
        raise LoadError(
            "No tools found in input. Please provide at least one tool entry.",
            source=source,
        )

    try:
        return SaasStack(tools=tools)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        messages = [
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in errors
        ]
        raise LoadError(
            f"Stack validation error: {'; '.join(messages)}",
            source=source,
        ) from exc


def load_interactive(tools_data: list[dict[str, object]]) -> SaasStack:
    """Build a SaasStack from a list of raw tool dicts gathered interactively.

    This function is intended for use by the interactive CLI command which
    collects tool information via prompts and passes raw dicts here.

    Args:
        tools_data: List of raw tool dictionaries with at minimum 'name'
            and 'category' keys.

    Returns:
        A validated SaasStack.

    Raises:
        LoadError: If any tool fails validation or the list is empty.
    """
    if not tools_data:
        raise LoadError(
            "No tools provided. Please enter at least one tool.",
            source="interactive",
        )

    tools: list[SaasTool] = []
    for idx, raw in enumerate(tools_data, start=1):
        tools.append(_coerce_tool_dict(raw, source="interactive", row=idx))

    try:
        return SaasStack(tools=tools)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        messages = [
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in errors
        ]
        raise LoadError(
            f"Stack validation error: {'; '.join(messages)}",
            source="interactive",
        ) from exc


def supported_categories() -> list[str]:
    """Return a sorted list of all supported tool category values.

    Returns:
        Sorted list of ToolCategory value strings.
    """
    return sorted(c.value for c in ToolCategory)
