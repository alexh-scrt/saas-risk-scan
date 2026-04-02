"""Unit tests for saas_risk_scan/knowledge_base.py.

Covers lookup, coverage counts, entry integrity, and category filtering.
"""

from __future__ import annotations

import pytest

from saas_risk_scan.knowledge_base import (
    KnowledgeBaseEntry,
    all_entries,
    entries_by_category,
    is_known,
    knowledge_base_size,
    known_tool_names,
    lookup,
)
from saas_risk_scan.models import ToolCategory


# ---------------------------------------------------------------------------
# knowledge_base_size
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSize:
    """Tests for knowledge_base_size()."""

    def test_at_least_50_entries(self) -> None:
        """Knowledge base should have at least 50 tools as specified."""
        assert knowledge_base_size() >= 50

    def test_consistent_with_all_entries(self) -> None:
        """Size should match the length of all_entries()."""
        assert knowledge_base_size() == len(all_entries())

    def test_consistent_with_known_names(self) -> None:
        """Size should match the number of known tool names."""
        assert knowledge_base_size() == len(known_tool_names())


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


class TestLookup:
    """Tests for the lookup() function."""

    def test_lookup_exact_case(self) -> None:
        """Lookup by exact canonical name should succeed."""
        entry = lookup("Zapier")
        assert entry is not None
        assert entry.name == "Zapier"

    def test_lookup_lowercase(self) -> None:
        """Lookup is case-insensitive."""
        entry = lookup("zapier")
        assert entry is not None
        assert entry.name == "Zapier"

    def test_lookup_uppercase(self) -> None:
        """Lookup by all-uppercase name."""
        entry = lookup("ZAPIER")
        assert entry is not None

    def test_lookup_mixed_case(self) -> None:
        """Lookup by mixed-case name."""
        entry = lookup("zApIeR")
        assert entry is not None

    def test_lookup_with_whitespace(self) -> None:
        """Lookup strips surrounding whitespace."""
        entry = lookup("  Zapier  ")
        assert entry is not None

    def test_lookup_unknown_tool_returns_none(self) -> None:
        """Unknown tool names return None."""
        entry = lookup("NonExistentTool12345")
        assert entry is None

    def test_lookup_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        entry = lookup("")
        assert entry is None

    def test_lookup_known_tools_spot_check(self) -> None:
        """Spot-check several well-known tools."""
        known = [
            "Salesforce", "HubSpot", "Notion", "Slack", "Jira",
            "Zendesk", "Intercom", "GitHub", "Stripe", "Datadog",
        ]
        for name in known:
            assert lookup(name) is not None, f"Expected '{name}' in knowledge base"


# ---------------------------------------------------------------------------
# is_known
# ---------------------------------------------------------------------------


class TestIsKnown:
    """Tests for the is_known() function."""

    def test_known_tool_returns_true(self) -> None:
        assert is_known("Zapier") is True

    def test_case_insensitive(self) -> None:
        assert is_known("zapier") is True
        assert is_known("ZAPIER") is True

    def test_unknown_tool_returns_false(self) -> None:
        assert is_known("RandomUnknownSaaS") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_known("") is False


# ---------------------------------------------------------------------------
# all_entries
# ---------------------------------------------------------------------------


class TestAllEntries:
    """Tests for the all_entries() function."""

    def test_returns_list(self) -> None:
        entries = all_entries()
        assert isinstance(entries, list)

    def test_all_entries_are_knowledge_base_entry(self) -> None:
        for entry in all_entries():
            assert isinstance(entry, KnowledgeBaseEntry)

    def test_returns_copy(self) -> None:
        """Modifying the returned list should not affect subsequent calls."""
        entries1 = all_entries()
        entries1.clear()
        entries2 = all_entries()
        assert len(entries2) > 0


# ---------------------------------------------------------------------------
# known_tool_names
# ---------------------------------------------------------------------------


class TestKnownToolNames:
    """Tests for the known_tool_names() function."""

    def test_returns_sorted_list(self) -> None:
        names = known_tool_names()
        assert names == sorted(names)

    def test_all_strings(self) -> None:
        for name in known_tool_names():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_contains_expected_tools(self) -> None:
        names_lower = {n.lower() for n in known_tool_names()}
        for expected in ["zapier", "salesforce", "notion", "slack", "github"]:
            assert expected in names_lower


# ---------------------------------------------------------------------------
# entries_by_category
# ---------------------------------------------------------------------------


class TestEntriesByCategory:
    """Tests for the entries_by_category() function."""

    def test_automation_category_has_entries(self) -> None:
        entries = entries_by_category(ToolCategory.AUTOMATION)
        assert len(entries) >= 2  # Zapier, Make, Calendly at minimum

    def test_crm_category_has_entries(self) -> None:
        entries = entries_by_category(ToolCategory.CRM)
        assert len(entries) >= 2

    def test_all_returned_entries_match_category(self) -> None:
        for category in ToolCategory:
            entries = entries_by_category(category)
            for entry in entries:
                assert entry.category == category

    def test_security_category_has_entries(self) -> None:
        entries = entries_by_category(ToolCategory.SECURITY)
        assert len(entries) >= 1

    def test_other_category_empty_or_has_entries(self) -> None:
        """'other' category may or may not have entries — just verify no crash."""
        entries = entries_by_category(ToolCategory.OTHER)
        assert isinstance(entries, list)

    def test_sum_of_categories_equals_total(self) -> None:
        """Entries partitioned by category should equal total size."""
        total = sum(
            len(entries_by_category(cat)) for cat in ToolCategory
        )
        assert total == knowledge_base_size()


# ---------------------------------------------------------------------------
# KnowledgeBaseEntry integrity
# ---------------------------------------------------------------------------


class TestKnowledgeBaseEntryIntegrity:
    """Tests that every entry in the knowledge base has valid data."""

    @pytest.fixture(scope="class")
    def entries(self) -> list[KnowledgeBaseEntry]:
        return all_entries()

    def test_all_names_non_empty(self, entries: list[KnowledgeBaseEntry]) -> None:
        for entry in entries:
            assert isinstance(entry.name, str) and len(entry.name.strip()) > 0, (
                f"Entry has empty name: {entry!r}"
            )

    def test_all_categories_valid(self, entries: list[KnowledgeBaseEntry]) -> None:
        valid_categories = {c for c in ToolCategory}
        for entry in entries:
            assert entry.category in valid_categories, (
                f"Entry '{entry.name}' has invalid category: {entry.category}"
            )

    def test_all_dimension_scores_in_range(self, entries: list[KnowledgeBaseEntry]) -> None:
        for entry in entries:
            dims = entry.dimensions()
            for dim_name, score in dims.items():
                assert 0.0 <= score <= 10.0, (
                    f"Entry '{entry.name}' has out-of-range score for '{dim_name}': {score}"
                )

    def test_all_alternatives_non_empty(self, entries: list[KnowledgeBaseEntry]) -> None:
        for entry in entries:
            assert isinstance(entry.alternatives, list), (
                f"Entry '{entry.name}' alternatives is not a list"
            )
            assert len(entry.alternatives) > 0, (
                f"Entry '{entry.name}' has no alternatives"
            )

    def test_all_rationales_non_empty(self, entries: list[KnowledgeBaseEntry]) -> None:
        for entry in entries:
            assert isinstance(entry.rationale, str) and len(entry.rationale.strip()) > 0, (
                f"Entry '{entry.name}' has empty rationale"
            )

    def test_dimensions_method_returns_five_keys(self, entries: list[KnowledgeBaseEntry]) -> None:
        expected_keys = {
            "task_automation_ratio",
            "api_openness",
            "workflow_complexity",
            "data_sensitivity",
            "incumbent_inertia",
        }
        for entry in entries:
            assert set(entry.dimensions().keys()) == expected_keys

    def test_no_duplicate_names_in_knowledge_base(self) -> None:
        names = [e.name.lower() for e in all_entries()]
        assert len(names) == len(set(names)), (
            "Knowledge base contains duplicate tool names"
        )


# ---------------------------------------------------------------------------
# Spot-check specific entries
# ---------------------------------------------------------------------------


class TestSpecificEntries:
    """Spot-checks for specific well-known tools."""

    def test_zapier_is_high_risk(self) -> None:
        """Zapier should have very high task automation ratio."""
        entry = lookup("Zapier")
        assert entry is not None
        assert entry.task_automation_ratio >= 8.0
        assert entry.category == ToolCategory.AUTOMATION

    def test_salesforce_has_high_inertia(self) -> None:
        """Salesforce should have high incumbent inertia."""
        entry = lookup("Salesforce")
        assert entry is not None
        assert entry.incumbent_inertia >= 8.0
        assert entry.category == ToolCategory.CRM

    def test_workday_has_high_data_sensitivity(self) -> None:
        """Workday should have very high data sensitivity (PII/payroll)."""
        entry = lookup("Workday")
        assert entry is not None
        assert entry.data_sensitivity >= 9.0
        assert entry.category == ToolCategory.HR

    def test_stripe_has_perfect_api_openness(self) -> None:
        """Stripe should have maximum API openness score."""
        entry = lookup("Stripe")
        assert entry is not None
        assert entry.api_openness == 10.0

    def test_calendly_has_good_alternatives(self) -> None:
        """Calendly should list Cal.com as an alternative."""
        entry = lookup("Calendly")
        assert entry is not None
        # At least one alternative should mention cal.com or scheduling
        alts_lower = [a.lower() for a in entry.alternatives]
        assert any("cal.com" in a for a in alts_lower)

    def test_notion_has_alternatives(self) -> None:
        """Notion should have multiple alternatives."""
        entry = lookup("Notion")
        assert entry is not None
        assert len(entry.alternatives) >= 2
