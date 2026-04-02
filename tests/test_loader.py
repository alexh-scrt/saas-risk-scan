"""Unit tests for saas_risk_scan/loader.py.

Covers YAML, JSON, and CSV loading, format detection, error handling,
and the interactive loader helper.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from saas_risk_scan.loader import (
    LoadError,
    detect_format,
    load_file,
    load_interactive,
    load_string,
    supported_categories,
)
from saas_risk_scan.models import SaasStack, SaasTool, ToolCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MINIMAL_YAML = textwrap.dedent("""\
    tools:
      - name: Zapier
        category: automation
""")

FULL_YAML = textwrap.dedent("""\
    tools:
      - name: Zapier
        category: automation
        monthly_cost_usd: 599
        team_size: 12
        notes: Test notes
      - name: Notion
        category: knowledge_management
        monthly_cost_usd: 320
        team_size: 80
""")

MINIMAL_JSON = json.dumps({"tools": [{"name": "Zapier", "category": "automation"}]})

FULL_JSON = json.dumps({
    "tools": [
        {
            "name": "Zapier",
            "category": "automation",
            "monthly_cost_usd": 599,
            "team_size": 12,
            "notes": "Test notes",
        },
        {
            "name": "Notion",
            "category": "knowledge_management",
            "monthly_cost_usd": 320,
            "team_size": 80,
        },
    ]
})

MINIMAL_CSV = "name,category\nZapier,automation\n"

FULL_CSV = textwrap.dedent("""\
    name,category,monthly_cost_usd,team_size,notes
    Zapier,automation,599,12,Test notes
    Notion,knowledge_management,320,80,
""")


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for the detect_format() function."""

    def test_yaml_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.touch()
        assert detect_format(p) == "yaml"

    def test_yml_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yml"
        p.touch()
        assert detect_format(p) == "yaml"

    def test_json_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.json"
        p.touch()
        assert detect_format(p) == "json"

    def test_csv_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.csv"
        p.touch()
        assert detect_format(p) == "csv"

    def test_uppercase_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "STACK.YAML"
        p.touch()
        assert detect_format(p) == "yaml"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.toml"
        p.touch()
        with pytest.raises(LoadError, match="Unsupported file extension"):
            detect_format(p)

    def test_no_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack"
        p.touch()
        with pytest.raises(LoadError):
            detect_format(p)


# ---------------------------------------------------------------------------
# load_string — YAML
# ---------------------------------------------------------------------------


class TestLoadStringYaml:
    """Tests for load_string() with YAML format."""

    def test_minimal_yaml(self) -> None:
        stack = load_string(MINIMAL_YAML, fmt="yaml")
        assert isinstance(stack, SaasStack)
        assert stack.tool_count() == 1
        assert stack.tools[0].name == "Zapier"

    def test_full_yaml(self) -> None:
        stack = load_string(FULL_YAML, fmt="yaml")
        assert stack.tool_count() == 2
        zapier = stack.tools[0]
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12
        assert zapier.notes == "Test notes"

    def test_bare_list_yaml(self) -> None:
        content = "- name: Zapier\n  category: automation\n"
        stack = load_string(content, fmt="yaml")
        assert stack.tool_count() == 1

    def test_case_insensitive_format_arg(self) -> None:
        stack = load_string(MINIMAL_YAML, fmt="YAML")
        assert stack.tool_count() == 1

    def test_yml_format_arg(self) -> None:
        stack = load_string(MINIMAL_YAML, fmt="yml")
        assert stack.tool_count() == 1

    def test_missing_tools_key_raises(self) -> None:
        content = "name: Zapier\ncategory: automation\n"
        with pytest.raises(LoadError, match="'tools'"):
            load_string(content, fmt="yaml")

    def test_empty_yaml_raises(self) -> None:
        with pytest.raises(LoadError):
            load_string("", fmt="yaml")

    def test_null_yaml_raises(self) -> None:
        with pytest.raises(LoadError):
            load_string("null\n", fmt="yaml")

    def test_invalid_yaml_raises(self) -> None:
        with pytest.raises(LoadError, match="YAML parse error"):
            load_string(": invalid: yaml: [", fmt="yaml")

    def test_missing_required_field_raises(self) -> None:
        content = "tools:\n  - name: Zapier\n"  # missing category
        with pytest.raises(LoadError, match="category"):
            load_string(content, fmt="yaml")

    def test_invalid_category_raises(self) -> None:
        content = "tools:\n  - name: Zapier\n    category: invalid_cat\n"
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_category_case_normalized(self) -> None:
        content = "tools:\n  - name: Zapier\n    category: AUTOMATION\n"
        stack = load_string(content, fmt="yaml")
        assert stack.tools[0].category == ToolCategory.AUTOMATION

    def test_tools_not_list_raises(self) -> None:
        content = "tools: not_a_list\n"
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_non_dict_entry_raises(self) -> None:
        content = "tools:\n  - just_a_string\n"
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_empty_tools_list_raises(self) -> None:
        content = "tools: []\n"
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_multiple_tools_parsed(self) -> None:
        content = textwrap.dedent("""\
            tools:
              - name: Zapier
                category: automation
              - name: Salesforce
                category: crm
              - name: Notion
                category: knowledge_management
        """)
        stack = load_string(content, fmt="yaml")
        assert stack.tool_count() == 3

    def test_optional_fields_default_to_none(self) -> None:
        stack = load_string(MINIMAL_YAML, fmt="yaml")
        tool = stack.tools[0]
        assert tool.monthly_cost_usd is None
        assert tool.team_size is None
        assert tool.notes is None


# ---------------------------------------------------------------------------
# load_string — JSON
# ---------------------------------------------------------------------------


class TestLoadStringJson:
    """Tests for load_string() with JSON format."""

    def test_minimal_json(self) -> None:
        stack = load_string(MINIMAL_JSON, fmt="json")
        assert stack.tool_count() == 1
        assert stack.tools[0].name == "Zapier"

    def test_full_json(self) -> None:
        stack = load_string(FULL_JSON, fmt="json")
        assert stack.tool_count() == 2

    def test_bare_array_json(self) -> None:
        content = json.dumps([{"name": "Zapier", "category": "automation"}])
        stack = load_string(content, fmt="json")
        assert stack.tool_count() == 1

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(LoadError, match="JSON parse error"):
            load_string("{not valid json", fmt="json")

    def test_missing_tools_key_raises(self) -> None:
        content = json.dumps({"name": "Zapier", "category": "automation"})
        with pytest.raises(LoadError, match="'tools'"):
            load_string(content, fmt="json")

    def test_empty_tools_list_raises(self) -> None:
        content = json.dumps({"tools": []})
        with pytest.raises(LoadError):
            load_string(content, fmt="json")

    def test_missing_required_field_raises(self) -> None:
        content = json.dumps({"tools": [{"name": "Zapier"}]})  # missing category
        with pytest.raises(LoadError, match="category"):
            load_string(content, fmt="json")

    def test_numeric_cost_parsed(self) -> None:
        content = json.dumps({
            "tools": [{"name": "Zapier", "category": "automation", "monthly_cost_usd": 599}]
        })
        stack = load_string(content, fmt="json")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_wrong_type_for_tools_raises(self) -> None:
        content = json.dumps({"tools": "not a list"})
        with pytest.raises(LoadError):
            load_string(content, fmt="json")


# ---------------------------------------------------------------------------
# load_string — CSV
# ---------------------------------------------------------------------------


class TestLoadStringCsv:
    """Tests for load_string() with CSV format."""

    def test_minimal_csv(self) -> None:
        stack = load_string(MINIMAL_CSV, fmt="csv")
        assert stack.tool_count() == 1
        assert stack.tools[0].name == "Zapier"

    def test_full_csv(self) -> None:
        stack = load_string(FULL_CSV, fmt="csv")
        assert stack.tool_count() == 2
        zapier = stack.tools[0]
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12
        assert zapier.notes == "Test notes"

    def test_empty_optional_fields_become_none(self) -> None:
        stack = load_string(FULL_CSV, fmt="csv")
        notion = stack.tools[1]
        assert notion.notes is None

    def test_missing_required_column_raises(self) -> None:
        content = "name,monthly_cost_usd\nZapier,599\n"  # missing category
        with pytest.raises(LoadError, match="category"):
            load_string(content, fmt="csv")

    def test_invalid_cost_raises(self) -> None:
        content = "name,category,monthly_cost_usd\nZapier,automation,not_a_number\n"
        with pytest.raises(LoadError, match="monthly_cost_usd"):
            load_string(content, fmt="csv")

    def test_invalid_team_size_raises(self) -> None:
        content = "name,category,team_size\nZapier,automation,not_an_int\n"
        with pytest.raises(LoadError, match="team_size"):
            load_string(content, fmt="csv")

    def test_column_alias_tool_name(self) -> None:
        """CSV 'tool_name' column should map to 'name'."""
        content = "tool_name,category\nZapier,automation\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].name == "Zapier"

    def test_column_alias_cost(self) -> None:
        """CSV 'cost' column should map to 'monthly_cost_usd'."""
        content = "name,category,cost\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_column_alias_seats(self) -> None:
        """CSV 'seats' column should map to 'team_size'."""
        content = "name,category,seats\nZapier,automation,50\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].team_size == 50

    def test_case_insensitive_headers(self) -> None:
        content = "Name,Category\nZapier,automation\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].name == "Zapier"

    def test_blank_rows_skipped(self) -> None:
        content = "name,category\nZapier,automation\n\n\n"
        stack = load_string(content, fmt="csv")
        assert stack.tool_count() == 1

    def test_multiple_tools(self) -> None:
        content = textwrap.dedent("""\
            name,category
            Zapier,automation
            Salesforce,crm
            Notion,knowledge_management
        """)
        stack = load_string(content, fmt="csv")
        assert stack.tool_count() == 3

    def test_team_size_float_string_coerced_to_int(self) -> None:
        """'50.0' in CSV should parse as team_size=50."""
        content = "name,category,team_size\nZapier,automation,50.0\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].team_size == 50


# ---------------------------------------------------------------------------
# load_string — error cases
# ---------------------------------------------------------------------------


class TestLoadStringErrors:
    """General error handling tests for load_string()."""

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(LoadError, match="Unknown format"):
            load_string(MINIMAL_YAML, fmt="xml")

    def test_load_error_includes_source(self) -> None:
        try:
            load_string("tools: []\n", fmt="yaml", source="myfile.yaml")
        except LoadError as exc:
            assert "myfile.yaml" in str(exc)


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
    """Tests for the load_file() function."""

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.write_text(FULL_YAML, encoding="utf-8")
        stack = load_file(p)
        assert stack.tool_count() == 2

    def test_load_yml_file(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yml"
        p.write_text(MINIMAL_YAML, encoding="utf-8")
        stack = load_file(p)
        assert stack.tool_count() == 1

    def test_load_json_file(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.json"
        p.write_text(FULL_JSON, encoding="utf-8")
        stack = load_file(p)
        assert stack.tool_count() == 2

    def test_load_csv_file(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.csv"
        p.write_text(FULL_CSV, encoding="utf-8")
        stack = load_file(p)
        assert stack.tool_count() == 2

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.write_text(MINIMAL_YAML, encoding="utf-8")
        stack = load_file(str(p))
        assert stack.tool_count() == 1

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(LoadError, match="not found"):
            load_file(p)

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError, match="not a file"):
            load_file(tmp_path)

    def test_load_sample_stack(self) -> None:
        """The bundled sample_stack.yaml should load without errors."""
        sample_path = Path("examples/sample_stack.yaml")
        if not sample_path.exists():
            pytest.skip("examples/sample_stack.yaml not found")
        stack = load_file(sample_path)
        assert stack.tool_count() >= 10


# ---------------------------------------------------------------------------
# load_interactive
# ---------------------------------------------------------------------------


class TestLoadInteractive:
    """Tests for the load_interactive() function."""

    def test_single_tool(self) -> None:
        data = [{"name": "Zapier", "category": "automation"}]
        stack = load_interactive(data)
        assert stack.tool_count() == 1
        assert stack.tools[0].name == "Zapier"

    def test_multiple_tools(self) -> None:
        data = [
            {"name": "Zapier", "category": "automation"},
            {"name": "Notion", "category": "knowledge_management"},
        ]
        stack = load_interactive(data)
        assert stack.tool_count() == 2

    def test_with_optional_fields(self) -> None:
        data = [{
            "name": "Salesforce",
            "category": "crm",
            "monthly_cost_usd": 3200.0,
            "team_size": 35,
            "notes": "Core CRM",
        }]
        stack = load_interactive(data)
        tool = stack.tools[0]
        assert tool.monthly_cost_usd == 3200.0
        assert tool.team_size == 35

    def test_empty_list_raises(self) -> None:
        with pytest.raises(LoadError, match="No tools provided"):
            load_interactive([])

    def test_missing_required_field_raises(self) -> None:
        data = [{"name": "Zapier"}]  # missing category
        with pytest.raises(LoadError, match="category"):
            load_interactive(data)

    def test_invalid_category_raises(self) -> None:
        data = [{"name": "Zapier", "category": "not_a_real_category"}]
        with pytest.raises(LoadError):
            load_interactive(data)

    def test_returns_saas_stack(self) -> None:
        data = [{"name": "Zapier", "category": "automation"}]
        result = load_interactive(data)
        assert isinstance(result, SaasStack)


# ---------------------------------------------------------------------------
# supported_categories
# ---------------------------------------------------------------------------


class TestSupportedCategories:
    """Tests for the supported_categories() function."""

    def test_returns_sorted_list(self) -> None:
        cats = supported_categories()
        assert cats == sorted(cats)

    def test_contains_automation(self) -> None:
        assert "automation" in supported_categories()

    def test_contains_all_tool_categories(self) -> None:
        cats = set(supported_categories())
        for tc in ToolCategory:
            assert tc.value in cats

    def test_all_strings(self) -> None:
        for cat in supported_categories():
            assert isinstance(cat, str)


# ---------------------------------------------------------------------------
# LoadError
# ---------------------------------------------------------------------------


class TestLoadError:
    """Tests for the LoadError exception class."""

    def test_basic_message(self) -> None:
        exc = LoadError("something went wrong")
        assert "something went wrong" in str(exc)

    def test_with_source(self) -> None:
        exc = LoadError("bad data", source="myfile.yaml")
        assert "myfile.yaml" in str(exc)
        assert exc.source == "myfile.yaml"

    def test_with_row(self) -> None:
        exc = LoadError("missing field", source="stack.csv", row=5)
        assert "row 5" in str(exc)
        assert exc.row == 5

    def test_message_attribute(self) -> None:
        exc = LoadError("test message", source="file.yaml", row=3)
        assert exc.message == "test message"

    def test_is_exception(self) -> None:
        exc = LoadError("test")
        assert isinstance(exc, Exception)
