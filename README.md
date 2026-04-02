# SaaS Risk Scan

> Audit your SaaS stack for AI displacement risk in minutes.

`saas-risk-scan` is a CLI tool that helps engineering and operations teams understand which SaaS tools in their stack are most likely to be replaced—or significantly disrupted—by AI agents and open-source alternatives. It scores each tool across five risk dimensions, ranks them, estimates replacement timelines, and suggests concrete agentic alternatives.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Commands](#commands)
- [Scoring Model](#scoring-model)
- [Sample Output](#sample-output)
- [LLM Enrichment](#llm-enrichment)
- [Configuration](#configuration)
- [Development](#development)

---

## Features

- **Multi-dimensional risk scoring** across 5 axes: automation ratio, API openness, workflow complexity, data lock-in, and market pressure — each weighted into a 0–100 displacement score.
- **Built-in knowledge base** covering 50+ common SaaS tools (Zapier, Notion, Salesforce, Intercom, etc.) with pre-researched baselines and curated agentic alternatives.
- **Estimated replacement timelines** — Near (<12 months), Mid (12–24 months), Long (24–36 months) — derived from score bands and category heuristics.
- **Optional LLM enrichment** using OpenAI to score unknown or niche tools automatically, with graceful fallback to prompt-based scoring.
- **Flexible I/O**: YAML, JSON, or CSV input; Rich terminal table, Markdown file, or JSON output.

---

## Installation

### From PyPI (once published)

```bash
pip install saas-risk-scan
```

### From source

```bash
git clone https://github.com/example/saas-risk-scan.git
cd saas-risk-scan
pip install -e .
```

### With optional LLM enrichment

```bash
pip install saas-risk-scan
export OPENAI_API_KEY="sk-..."
```

---

## Quick Start

```bash
# Scan from a YAML file and display a Rich terminal table
saas-risk-scan scan examples/sample_stack.yaml

# Scan and export a Markdown report
saas-risk-scan scan examples/sample_stack.yaml --output report.md --format markdown

# Scan and export JSON
saas-risk-scan scan examples/sample_stack.yaml --output results.json --format json

# Interactive mode — prompts you to enter tools one by one
saas-risk-scan interactive

# Export a previously generated JSON result to Markdown
saas-risk-scan export results.json --format markdown
```

---

## Input Format

The tool accepts **YAML**, **JSON**, or **CSV** files listing your SaaS stack.

### YAML (recommended)

```yaml
tools:
  - name: Zapier
    category: automation
    monthly_cost_usd: 599
    team_size: 50
    notes: "Used for 200+ zaps across marketing and ops"

  - name: Intercom
    category: customer_support
    monthly_cost_usd: 1200
    team_size: 10

  - name: Notion
    category: knowledge_management
    monthly_cost_usd: 160
    team_size: 80
    notes: "Company wiki and project tracking"
```

### JSON

```json
{
  "tools": [
    {
      "name": "Zapier",
      "category": "automation",
      "monthly_cost_usd": 599,
      "team_size": 50
    }
  ]
}
```

### CSV

```csv
name,category,monthly_cost_usd,team_size,notes
Zapier,automation,599,50,Used for 200+ zaps
Intercom,customer_support,1200,10,
Notion,knowledge_management,160,80,Company wiki
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Name of the SaaS tool |
| `category` | string | ✅ | Tool category (see categories below) |
| `monthly_cost_usd` | number | ❌ | Monthly spend in USD |
| `team_size` | integer | ❌ | Number of users/seats |
| `notes` | string | ❌ | Free-text notes for context |

### Supported Categories

`automation`, `crm`, `customer_support`, `data_analytics`, `knowledge_management`,
`project_management`, `communication`, `hr`, `finance`, `marketing`, `devtools`,
`security`, `storage`, `ecommerce`, `other`

---

## Commands

### `scan`

Scan a file of SaaS tools and produce a risk report.

```
saas-risk-scan scan [INPUT_FILE] [OPTIONS]

Arguments:
  INPUT_FILE  Path to YAML, JSON, or CSV file  [required]

Options:
  --output    -o  Output file path (default: stdout)
  --format    -f  Output format: table|markdown|json  [default: table]
  --enrich    -e  Enable LLM enrichment for unknown tools  [flag]
  --top       -n  Show only top N tools by risk score
  --min-score     Filter: only show tools with score >= this value
  --help          Show this message and exit.
```

### `interactive`

Enter tools interactively via prompts.

```
saas-risk-scan interactive [OPTIONS]

Options:
  --output    -o  Output file path (default: stdout)
  --format    -f  Output format: table|markdown|json  [default: table]
  --enrich    -e  Enable LLM enrichment  [flag]
  --help          Show this message and exit.
```

### `export`

Convert a saved JSON result to another format.

```
saas-risk-scan export [INPUT_JSON] [OPTIONS]

Arguments:
  INPUT_JSON  Path to a JSON results file

Options:
  --output    -o  Output file path  [required]
  --format    -f  Output format: markdown|json  [default: markdown]
  --help          Show this message and exit.
```

---

## Scoring Model

Each tool is scored on five dimensions (0–10 each), then combined into a weighted **Displacement Score** (0–100):

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Task Automation Ratio** | 30% | How much of the tool's core value is pure task execution vs. judgment |
| **API Openness** | 20% | Quality and completeness of public APIs / webhook support |
| **Workflow Complexity** | 15% | How deeply the tool is embedded in multi-step human workflows |
| **Data Sensitivity** | 20% | Risk and friction of data migration / lock-in |
| **Incumbent Inertia** | 15% | Organizational switching cost (contracts, training, political capital) |

### Score Bands → Replacement Timeline

| Score | Risk Level | Estimated Timeline |
|-------|------------|--------------------|
| 75–100 | 🔴 Critical | Near-term: < 12 months |
| 50–74 | 🟠 High | Mid-term: 12–24 months |
| 25–49 | 🟡 Medium | Long-term: 24–36 months |
| 0–24 | 🟢 Low | Unlikely / 36+ months |

---

## Sample Output

### Terminal (Rich table)

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           SaaS Displacement Risk Report                                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  Tool            Category        Score   Risk      Timeline      Agentic Alternatives
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  Zapier          automation        82   🔴 Critical  < 12 mo    n8n, LangChain agents, AutoGen
  Intercom        customer_support  71   🟠 High      12–24 mo   OpenAgents, custom LLM chatbot
  Notion          knowledge_mgmt    58   🟠 High      12–24 mo   Obsidian + AI plugins, Mem.ai
  Salesforce      crm               44   🟡 Medium    24–36 mo   Twenty CRM, AI-native pipelines
  Workday         hr                31   🟡 Medium    24–36 mo   Rippling + AI workflows
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  5 tools analyzed · Total monthly spend: $3,459 · Avg risk score: 57.2
```

### Markdown report excerpt

```markdown
# SaaS Displacement Risk Report

Generated: 2024-01-15 · Tools analyzed: 5 · Avg score: 57.2

## Executive Summary

1 tool is at **Critical** risk of near-term AI displacement (< 12 months).
2 tools face **High** displacement risk within 12–24 months.
2 tools face **Medium** risk over a 24–36 month horizon.

## Risk Rankings

| Rank | Tool | Category | Score | Risk | Timeline | Top Alternative |
|------|------|----------|-------|------|----------|-----------------|
| 1 | Zapier | automation | 82 | 🔴 Critical | < 12 mo | n8n + LangChain |
...
```

---

## LLM Enrichment

When you pass `--enrich`, the tool will use OpenAI's API to:

1. Score **unknown tools** not in the built-in knowledge base
2. Generate **agentic alternative suggestions** tailored to your use case
3. Provide **rationale text** explaining each risk score

```bash
export OPENAI_API_KEY="sk-..."
saas-risk-scan scan my_stack.yaml --enrich
```

Without an API key, the tool falls back gracefully:
- Known tools use built-in knowledge base scores
- Unknown tools use conservative defaults with a warning

**Model used**: `gpt-4o-mini` (cost-efficient; ~$0.001–0.01 per tool)

---

## Configuration

Set environment variables to configure behavior:

```bash
# Required for LLM enrichment
export OPENAI_API_KEY="sk-..."

# Optional: override OpenAI model
export SAAS_RISK_OPENAI_MODEL="gpt-4o-mini"

# Optional: set default output format
export SAAS_RISK_DEFAULT_FORMAT="markdown"
```

---

## Development

### Setup

```bash
git clone https://github.com/example/saas-risk-scan.git
cd saas-risk-scan
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
pytest tests/test_scorer.py -v
```

### Project structure

```
saas_risk_scan/
├── __init__.py        # Package init, version
├── main.py            # Typer CLI entry point
├── models.py          # Pydantic data models
├── scorer.py          # Rule-based scoring engine
├── knowledge_base.py  # Static tool baselines
├── enricher.py        # OpenAI enrichment
├── reporter.py        # Report rendering
└── loader.py          # File input loader
templates/
└── report.md.j2       # Jinja2 Markdown template
examples/
└── sample_stack.yaml  # Example input
tests/
├── test_scorer.py
├── test_loader.py
└── test_reporter.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
