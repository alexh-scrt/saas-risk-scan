# SaaS Risk Scan

> Know which SaaS tools AI will replace before it happens.

`saas-risk-scan` is a CLI tool that audits your SaaS stack for AI displacement risk. It scores each tool across five risk dimensions, ranks them by likelihood of replacement, estimates timelines (6–36 months), and suggests concrete agentic alternatives—all in minutes, from a single YAML file.

---

## Quick Start

```bash
# Install
pip install saas-risk-scan

# Scan a YAML file (outputs a Rich table to the terminal)
saas-risk-scan scan examples/sample_stack.yaml

# Export a Markdown report
saas-risk-scan scan examples/sample_stack.yaml --format markdown --output report.md

# Enable LLM enrichment for unknown tools
export OPENAI_API_KEY=sk-...
saas-risk-scan scan examples/sample_stack.yaml --enrich
```

That's it. Your ranked displacement risk report is ready.

---

## What It Does

`saas-risk-scan` takes a list of SaaS tools you're currently paying for and runs each one through a transparent, rule-based scoring model across five risk dimensions. It produces a ranked report showing which tools are most exposed to AI-driven replacement, estimated replacement windows, and specific open-source or agentic alternatives to consider. The scoring model works fully offline; OpenAI enrichment is optional for tools not in the built-in knowledge base.

---

## Features

- **5-axis risk scoring** — automation ratio, API openness, workflow complexity, data sensitivity, and market pressure, weighted into a single 0–100 displacement score per tool.
- **Built-in knowledge base** — 50+ common SaaS tools (Zapier, Notion, Salesforce, Intercom, and more) with pre-researched baselines and curated agentic alternatives (AutoGen, LangChain agents, open-source replacements).
- **Estimated replacement timelines** — Near (<12 mo), Mid (12–24 mo), and Long (24–36 mo) bands derived from score ranges and category heuristics.
- **Optional LLM enrichment** — Automatically scores unknown or niche tools via OpenAI (`gpt-4o-mini`), with graceful fallback to heuristic defaults when no API key is set.
- **Flexible I/O** — Accepts YAML, JSON, or CSV input; outputs a Rich terminal table, Markdown report, or structured JSON that can be committed to a repo or piped into other tools.

---

## Usage Examples

### Scan from a YAML file

```bash
saas-risk-scan scan my_stack.yaml
```

### Save a Markdown report

```bash
saas-risk-scan scan my_stack.yaml --format markdown --output risk-report.md
```

### Save a JSON report for further processing

```bash
saas-risk-scan scan my_stack.yaml --format json --output risk-report.json
```

### Convert a saved JSON report to Markdown

```bash
saas-risk-scan export risk-report.json --format markdown --output risk-report.md
```

### Run interactively (no input file needed)

```bash
saas-risk-scan interactive
```

### Filter results to high-risk tools only

```bash
saas-risk-scan scan my_stack.yaml --min-risk high
```

---

## Input Format

Create a YAML file with a top-level `tools` key. Only `name` and `category` are required; all other fields improve scoring accuracy.

```yaml
# my_stack.yaml
tools:
  - name: Zapier
    category: automation
    monthly_cost_usd: 599
    team_size: 12
    notes: "Handles lead routing and Slack notifications via 300+ zaps."

  - name: Notion
    category: knowledge_management
    monthly_cost_usd: 160
    team_size: 40

  - name: Intercom
    category: customer_support
    monthly_cost_usd: 1200
    team_size: 8
    notes: "Primary customer chat and support ticket system."
```

**Supported categories:** `automation`, `crm`, `customer_support`, `analytics`, `knowledge_management`, `project_management`, `hr`, `finance`, `devtools`, `marketing`, `security`, `other`

JSON and CSV formats are also supported. For CSV, use column headers matching the field names above.

---

## Sample Output

```
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool       ┃ Score    ┃ Risk Level            ┃ Timeline   ┃ Top Alternative                 ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Zapier     │ 87/100   │ 🔴 CRITICAL           │ <12 months │ LangChain agents / n8n          │
│ Intercom   │ 74/100   │ 🟠 HIGH               │ 12–24 mo   │ Open-source LLM support agent   │
│ Notion     │ 61/100   │ 🟡 MEDIUM             │ 12–24 mo   │ Outline + AI summarization      │
│ Salesforce │ 45/100   │ 🟢 LOW                │ 24–36 mo   │ Twenty CRM / HubSpot OSS        │
└────────────┴──────────┴───────────────────────┴────────────┴─────────────────────────────────┘
```

---

## Project Structure

```
saas-risk-scan/
├── pyproject.toml                  # Project metadata, deps, CLI entry point
├── examples/
│   └── sample_stack.yaml           # Example input for a realistic SaaS stack
├── templates/
│   └── report.md.j2                # Jinja2 template for Markdown reports
├── saas_risk_scan/
│   ├── __init__.py                 # Package init, version constant
│   ├── main.py                     # Typer CLI: scan, interactive, export commands
│   ├── models.py                   # Pydantic v2 models (SaasTool, RiskScore, etc.)
│   ├── scorer.py                   # Rule-based scoring engine (5 dimensions)
│   ├── knowledge_base.py           # Pre-researched baselines for 50+ SaaS tools
│   ├── enricher.py                 # Optional OpenAI enrichment for unknown tools
│   ├── loader.py                   # YAML / JSON / CSV input parsing & validation
│   └── reporter.py                 # Rich table, Markdown, and JSON output rendering
└── tests/
    ├── test_models.py
    ├── test_scorer.py
    ├── test_loader.py
    ├── test_reporter.py
    └── test_knowledge_base.py
```

---

## Configuration

All configuration is done via environment variables. No config file is required.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(unset)* | Enables LLM enrichment for unknown tools. Optional. |
| `SAAS_RISK_OPENAI_MODEL` | `gpt-4o-mini` | Override the OpenAI model used for enrichment. |
| `SAAS_RISK_DEFAULT_FORMAT` | `table` | Default output format (`table`, `markdown`, `json`). |

### CLI Options (all commands)

```
Options:
  --format    [table|markdown|json]  Output format (default: table)
  --output    PATH                   Write output to a file instead of stdout
  --enrich                           Enable OpenAI LLM enrichment
  --min-risk  [low|medium|high|critical]  Filter results to this risk level and above
  --top       INTEGER                Show only the top N tools by displacement score
  --help                             Show this message and exit
```

---

## Development

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/your-org/saas-risk-scan.git
cd saas-risk-scan
pip install -e ".[dev]"

# Run tests
pytest

# Run the CLI locally
python -m saas_risk_scan.main scan examples/sample_stack.yaml
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
