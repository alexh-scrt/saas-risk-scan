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
# Fixtures / shared content strings
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

    def test_uppercase_yaml_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "STACK.YAML"
        p.touch()
        assert detect_format(p) == "yaml"

    def test_uppercase_json_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "STACK.JSON"
        p.touch()
        assert detect_format(p) == "json"

    def test_uppercase_csv_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "STACK.CSV"
        p.touch()
        assert detect_format(p) == "csv"

    def test_mixed_case_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.Yaml"
        p.touch()
        assert detect_format(p) == "yaml"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.toml"
        p.touch()
        with pytest.raises(LoadError, match="Unsupported file extension"):
            detect_format(p)

    def test_unsupported_xml_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.xml"
        p.touch()
        with pytest.raises(LoadError):
            detect_format(p)

    def test_no_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack"
        p.touch()
        with pytest.raises(LoadError):
            detect_format(p)

    def test_returns_string(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.touch()
        result = detect_format(p)
        assert isinstance(result, str)


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

    def test_second_tool_parsed(self) -> None:
        stack = load_string(FULL_YAML, fmt="yaml")
        notion = stack.tools[1]
        assert notion.name == "Notion"
        assert notion.category == ToolCategory.KNOWLEDGE_MANAGEMENT
        assert notion.monthly_cost_usd == 320.0
        assert notion.team_size == 80

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

    def test_invalid_yaml_syntax_raises(self) -> None:
        with pytest.raises(LoadError, match="YAML parse error"):
            load_string(": invalid: yaml: [", fmt="yaml")

    def test_missing_required_field_name_raises(self) -> None:
        content = "tools:\n  - category: automation\n"  # missing name
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_missing_required_field_category_raises(self) -> None:
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

    def test_returns_saas_stack_instance(self) -> None:
        stack = load_string(MINIMAL_YAML, fmt="yaml")
        assert isinstance(stack, SaasStack)

    def test_tool_names_preserved(self) -> None:
        stack = load_string(FULL_YAML, fmt="yaml")
        names = [t.name for t in stack.tools]
        assert "Zapier" in names
        assert "Notion" in names

    def test_whitespace_stripped_from_name(self) -> None:
        content = "tools:\n  - name: '  Zapier  '\n    category: automation\n"
        stack = load_string(content, fmt="yaml")
        assert stack.tools[0].name == "Zapier"

    def test_negative_cost_raises(self) -> None:
        content = textwrap.dedent("""\
            tools:
              - name: Zapier
                category: automation
                monthly_cost_usd: -100
        """)
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_zero_team_size_raises(self) -> None:
        content = textwrap.dedent("""\
            tools:
              - name: Zapier
                category: automation
                team_size: 0
        """)
        with pytest.raises(LoadError):
            load_string(content, fmt="yaml")

    def test_source_label_in_error_message(self) -> None:
        content = "tools:\n  - name: Zapier\n"  # missing category
        try:
            load_string(content, fmt="yaml", source="myfile.yaml")
        except LoadError as exc:
            assert "myfile.yaml" in str(exc)

    def test_all_tool_categories_accepted(self) -> None:
        for cat in ToolCategory:
            content = f"tools:\n  - name: TestTool\n    category: {cat.value}\n"
            stack = load_string(content, fmt="yaml")
            assert stack.tools[0].category == cat


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

    def test_full_json_first_tool(self) -> None:
        stack = load_string(FULL_JSON, fmt="json")
        zapier = stack.tools[0]
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12
        assert zapier.notes == "Test notes"

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

    def test_missing_required_field_category_raises(self) -> None:
        content = json.dumps({"tools": [{"name": "Zapier"}]})  # missing category
        with pytest.raises(LoadError, match="category"):
            load_string(content, fmt="json")

    def test_missing_required_field_name_raises(self) -> None:
        content = json.dumps({"tools": [{"category": "automation"}]})  # missing name
        with pytest.raises(LoadError):
            load_string(content, fmt="json")

    def test_numeric_cost_parsed(self) -> None:
        content = json.dumps({
            "tools": [{"name": "Zapier", "category": "automation", "monthly_cost_usd": 599}]
        })
        stack = load_string(content, fmt="json")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_float_cost_parsed(self) -> None:
        content = json.dumps({
            "tools": [{"name": "Zapier", "category": "automation", "monthly_cost_usd": 599.99}]
        })
        stack = load_string(content, fmt="json")
        assert stack.tools[0].monthly_cost_usd == 599.99

    def test_wrong_type_for_tools_raises(self) -> None:
        content = json.dumps({"tools": "not a list"})
        with pytest.raises(LoadError):
            load_string(content, fmt="json")

    def test_non_dict_entry_raises(self) -> None:
        content = json.dumps({"tools": ["just a string"]})
        with pytest.raises(LoadError):
            load_string(content, fmt="json")

    def test_json_format_case_insensitive(self) -> None:
        stack = load_string(MINIMAL_JSON, fmt="JSON")
        assert stack.tool_count() == 1

    def test_category_normalized_from_json(self) -> None:
        content = json.dumps({"tools": [{"name": "Zapier", "category": "AUTOMATION"}]})
        stack = load_string(content, fmt="json")
        assert stack.tools[0].category == ToolCategory.AUTOMATION

    def test_multiple_tools(self) -> None:
        tools = [
            {"name": "A", "category": "automation"},
            {"name": "B", "category": "crm"},
            {"name": "C", "category": "hr"},
        ]
        content = json.dumps({"tools": tools})
        stack = load_string(content, fmt="json")
        assert stack.tool_count() == 3

    def test_optional_fields_none_by_default(self) -> None:
        stack = load_string(MINIMAL_JSON, fmt="json")
        tool = stack.tools[0]
        assert tool.monthly_cost_usd is None
        assert tool.team_size is None
        assert tool.notes is None

    def test_returns_saas_stack(self) -> None:
        stack = load_string(MINIMAL_JSON, fmt="json")
        assert isinstance(stack, SaasStack)

    def test_invalid_category_raises(self) -> None:
        content = json.dumps({"tools": [{"name": "Zapier", "category": "not_valid"}]})
        with pytest.raises(LoadError):
            load_string(content, fmt="json")

    def test_negative_cost_raises(self) -> None:
        content = json.dumps(
            {"tools": [{"name": "Zapier", "category": "automation", "monthly_cost_usd": -50}]}
        )
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

    def test_missing_required_column_category_raises(self) -> None:
        content = "name,monthly_cost_usd\nZapier,599\n"  # missing category
        with pytest.raises(LoadError, match="category"):
            load_string(content, fmt="csv")

    def test_missing_required_column_name_raises(self) -> None:
        content = "category,monthly_cost_usd\nautomation,599\n"  # missing name
        with pytest.raises(LoadError, match="name"):
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

    def test_column_alias_tool(self) -> None:
        """CSV 'tool' column should map to 'name'."""
        content = "tool,category\nZapier,automation\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].name == "Zapier"

    def test_column_alias_cost(self) -> None:
        """CSV 'cost' column should map to 'monthly_cost_usd'."""
        content = "name,category,cost\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_column_alias_monthly_cost(self) -> None:
        """CSV 'monthly_cost' column should map to 'monthly_cost_usd'."""
        content = "name,category,monthly_cost\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_column_alias_cost_usd(self) -> None:
        """CSV 'cost_usd' column should map to 'monthly_cost_usd'."""
        content = "name,category,cost_usd\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_column_alias_seats(self) -> None:
        """CSV 'seats' column should map to 'team_size'."""
        content = "name,category,seats\nZapier,automation,50\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].team_size == 50

    def test_column_alias_users(self) -> None:
        """CSV 'users' column should map to 'team_size'."""
        content = "name,category,users\nZapier,automation,50\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].team_size == 50

    def test_column_alias_size(self) -> None:
        """CSV 'size' column should map to 'team_size'."""
        content = "name,category,size\nZapier,automation,50\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].team_size == 50

    def test_column_alias_note(self) -> None:
        """CSV 'note' column should map to 'notes'."""
        content = "name,category,note\nZapier,automation,some note\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].notes == "some note"

    def test_column_alias_description(self) -> None:
        """CSV 'description' column should map to 'notes'."""
        content = "name,category,description\nZapier,automation,a desc\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].notes == "a desc"

    def test_case_insensitive_headers(self) -> None:
        content = "Name,Category\nZapier,automation\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].name == "Zapier"

    def test_header_with_spaces_normalized(self) -> None:
        content = "name,category,monthly cost usd\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        # 'monthly cost usd' normalizes to 'monthly_cost_usd'
        assert stack.tools[0].monthly_cost_usd == 599.0

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

    def test_returns_saas_stack(self) -> None:
        stack = load_string(MINIMAL_CSV, fmt="csv")
        assert isinstance(stack, SaasStack)

    def test_csv_format_case_insensitive(self) -> None:
        stack = load_string(MINIMAL_CSV, fmt="CSV")
        assert stack.tool_count() == 1

    def test_tools_are_saas_tool_instances(self) -> None:
        stack = load_string(FULL_CSV, fmt="csv")
        for tool in stack.tools:
            assert isinstance(tool, SaasTool)

    def test_category_normalized_lowercase(self) -> None:
        content = "name,category\nZapier,AUTOMATION\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].category == ToolCategory.AUTOMATION

    def test_zero_cost_parsed(self) -> None:
        content = "name,category,monthly_cost_usd\nStripe,finance,0\n"
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].monthly_cost_usd == 0.0

    def test_header_with_dashes_normalized(self) -> None:
        content = "name,category,monthly-cost-usd\nZapier,automation,599\n"
        stack = load_string(content, fmt="csv")
        # 'monthly-cost-usd' normalizes to 'monthly_cost_usd'
        assert stack.tools[0].monthly_cost_usd == 599.0

    def test_notes_with_comma_handled(self) -> None:
        """CSV with quoted notes containing commas should parse correctly."""
        content = 'name,category,notes\nZapier,automation,"Used for marketing, sales"\n'
        stack = load_string(content, fmt="csv")
        assert stack.tools[0].notes == "Used for marketing, sales"

    def test_all_only_blank_rows_raises(self) -> None:
        """If all data rows are blank, no tools are found."""
        content = "name,category\n\n\n"
        with pytest.raises(LoadError):
            load_string(content, fmt="csv")


# ---------------------------------------------------------------------------
# load_string — general error cases
# ---------------------------------------------------------------------------


class TestLoadStringErrors:
    """General error handling tests for load_string()."""

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(LoadError, match="Unknown format"):
            load_string(MINIMAL_YAML, fmt="xml")

    def test_unknown_format_toml_raises(self) -> None:
        with pytest.raises(LoadError):
            load_string("", fmt="toml")

    def test_load_error_includes_source(self) -> None:
        try:
            load_string("tools: []\n", fmt="yaml", source="myfile.yaml")
        except LoadError as exc:
            assert "myfile.yaml" in str(exc)

    def test_load_error_includes_row_for_csv(self) -> None:
        content = "name,category\nZapier,automation\nBadTool,invalid_cat\n"
        try:
            load_string(content, fmt="csv")
        except LoadError as exc:
            # Row 3 (data row 2) should be in the error
            assert exc.row is not None

    def test_no_tools_raises_with_helpful_message(self) -> None:
        content = "tools: []\n"
        try:
            load_string(content, fmt="yaml")
        except LoadError as exc:
            assert "tool" in exc.message.lower() or "empty" in exc.message.lower()

    def test_format_whitespace_stripped(self) -> None:
        """Format string with surrounding whitespace should still work."""
        stack = load_string(MINIMAL_YAML, fmt=" yaml ")
        assert stack.tool_count() == 1


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

    def test_returns_saas_stack(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.write_text(MINIMAL_YAML, encoding="utf-8")
        result = load_file(p)
        assert isinstance(result, SaasStack)

    def test_load_yaml_parses_all_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.yaml"
        p.write_text(FULL_YAML, encoding="utf-8")
        stack = load_file(p)
        zapier = stack.tools[0]
        assert zapier.name == "Zapier"
        assert zapier.category == ToolCategory.AUTOMATION
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12
        assert zapier.notes == "Test notes"

    def test_load_json_parses_all_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.json"
        p.write_text(FULL_JSON, encoding="utf-8")
        stack = load_file(p)
        zapier = stack.tools[0]
        assert zapier.name == "Zapier"
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12

    def test_load_csv_parses_all_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.csv"
        p.write_text(FULL_CSV, encoding="utf-8")
        stack = load_file(p)
        zapier = stack.tools[0]
        assert zapier.name == "Zapier"
        assert zapier.monthly_cost_usd == 599.0
        assert zapier.team_size == 12

    def test_load_sample_stack(self) -> None:
        """The bundled sample_stack.yaml should load without errors."""
        sample_path = Path("examples/sample_stack.yaml")
        if not sample_path.exists():
            pytest.skip("examples/sample_stack.yaml not found")
        stack = load_file(sample_path)
        assert stack.tool_count() >= 10

    def test_load_sample_stack_has_expected_tools(self) -> None:
        """The sample stack should contain well-known tools."""
        sample_path = Path("examples/sample_stack.yaml")
        if not sample_path.exists():
            pytest.skip("examples/sample_stack.yaml not found")
        stack = load_file(sample_path)
        names = {t.name.lower() for t in stack.tools}
        assert any("zapier" in n for n in names)
        assert any("salesforce" in n for n in names)

    def test_invalid_content_raises_load_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("tools:\n  - just_a_string\n", encoding="utf-8")
        with pytest.raises(LoadError):
            load_file(p)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "stack.toml"
        p.write_text("[tools]\n", encoding="utf-8")
        with pytest.raises(LoadError):
            load_file(p)


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
        assert tool.notes == "Core CRM"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(LoadError, match="No tools provided"):
            load_interactive([])

    def test_missing_required_field_raises(self) -> None:
        data = [{"name": "Zapier"}]  # missing category
        with pytest.raises(LoadError, match="category"):
            load_interactive(data)

    def test_missing_name_raises(self) -> None:
        data = [{"category": "automation"}]  # missing name
        with pytest.raises(LoadError):
            load_interactive(data)

    def test_invalid_category_raises(self) -> None:
        data = [{"name": "Zapier", "category": "not_a_real_category"}]
        with pytest.raises(LoadError):
            load_interactive(data)

    def test_returns_saas_stack(self) -> None:
        data = [{"name": "Zapier", "category": "automation"}]
        result = load_interactive(data)
        assert isinstance(result, SaasStack)

    def test_category_normalized_lowercase(self) -> None:
        data = [{"name": "Zapier", "category": "AUTOMATION"}]
        stack = load_interactive(data)
        assert stack.tools[0].category == ToolCategory.AUTOMATION

    def test_tools_are_saas_tool_instances(self) -> None:
        data = [
            {"name": "Zapier", "category": "automation"},
            {"name": "Notion", "category": "knowledge_management"},
        ]
        stack = load_interactive(data)
        for tool in stack.tools:
            assert isinstance(tool, SaasTool)

    def test_negative_cost_raises(self) -> None:
        data = [{"name": "Zapier", "category": "automation", "monthly_cost_usd": -10.0}]
        with pytest.raises(LoadError):
            load_interactive(data)

    def test_zero_team_size_raises(self) -> None:
        data = [{"name": "Zapier", "category": "automation", "team_size": 0}]
        with pytest.raises(LoadError):
            load_interactive(data)

    def test_preserves_order(self) -> None:
        data = [
            {"name": "Alpha", "category": "automation"},
            {"name": "Beta", "category": "crm"},
            {"name": "Gamma", "category": "hr"},
        ]
        stack = load_interactive(data)
        assert stack.tools[0].name == "Alpha"
        assert stack.tools[1].name == "Beta"
        assert stack.tools[2].name == "Gamma"

    def test_unknown_fields_ignored(self) -> None:
        """Extra unknown fields should be silently ignored."""
        data = [{
            "name": "Zapier",
            "category": "automation",
            "unknown_field": "some_value",
            "another_field": 42,
        }]
        stack = load_interactive(data)
        assert stack.tool_count() == 1
        assert stack.tools[0].name == "Zapier"

    def test_error_includes_row_number(self) -> None:
        """Error for a specific row should include row context."""
        data = [
            {"name": "Zapier", "category": "automation"},
            {"name": "BadTool", "category": "not_valid"},  # row 2
        ]
        try:
            load_interactive(data)
        except LoadError as exc:
            assert exc.row is not None

    def test_all_categories_accepted(self) -> None:
        for cat in ToolCategory:
            data = [{"name": "TestTool", "category": cat.value}]
            stack = load_interactive(data)
            assert stack.tools[0].category == cat


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

    def test_contains_crm(self) -> None:
        assert "crm" in supported_categories()

    def test_contains_hr(self) -> None:
        assert "hr" in supported_categories()

    def test_contains_finance(self) -> None:
        assert "finance" in supported_categories()

    def test_contains_other(self) -> None:
        assert "other" in supported_categories()

    def test_contains_all_tool_categories(self) -> None:
        cats = set(supported_categories())
        for tc in ToolCategory:
            assert tc.value in cats

    def test_all_strings(self) -> None:
        for cat in supported_categories():
            assert isinstance(cat, str)

    def test_no_duplicates(self) -> None:
        cats = supported_categories()
        assert len(cats) == len(set(cats))

    def test_returns_list(self) -> None:
        cats = supported_categories()
        assert isinstance(cats, list)

    def test_count_matches_tool_category_enum(self) -> None:
        cats = supported_categories()
        assert len(cats) == len(ToolCategory)


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

    def test_source_attribute_none_by_default(self) -> None:
        exc = LoadError("test message")
        assert exc.source is None

    def test_row_attribute_none_by_default(self) -> None:
        exc = LoadError("test message")
        assert exc.row is None

    def test_row_attribute_none_with_source(self) -> None:
        exc = LoadError("test message", source="file.yaml")
        assert exc.row is None

    def test_str_representation_includes_all_parts(self) -> None:
        exc = LoadError("missing field", source="data.csv", row=7)
        s = str(exc)
        assert "data.csv" in s
        assert "row 7" in s
        assert "missing field" in s

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LoadError) as exc_info:
            raise LoadError("test error", source="test.yaml", row=1)
        assert exc_info.value.message == "test error"
        assert exc_info.value.source == "test.yaml"
        assert exc_info.value.row == 1

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(LoadError, Exception)
