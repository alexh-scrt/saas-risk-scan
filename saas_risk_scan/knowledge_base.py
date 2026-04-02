"""Static knowledge base mapping known SaaS tools to baseline scores, categories, and agentic alternatives.

This module provides pre-researched risk dimension scores and curated agentic
alternatives for 50+ common SaaS tools. The knowledge base is used by the
scoring engine to avoid needing LLM enrichment for well-known tools.

Each entry contains:
    - name: canonical tool name (lowercase for matching)
    - category: ToolCategory value
    - dimensions: dict of the five risk dimension scores (0–10)
    - alternatives: list of agentic/open-source replacement suggestions
    - rationale: brief human-readable explanation of the scores
"""

from __future__ import annotations

from typing import Optional

from saas_risk_scan.models import ToolCategory


# ---------------------------------------------------------------------------
# Type alias for a knowledge base entry
# ---------------------------------------------------------------------------

class KnowledgeBaseEntry:
    """Represents a single pre-researched SaaS tool entry in the knowledge base.

    Attributes:
        name: Canonical tool name (display form).
        category: Tool category.
        task_automation_ratio: How automatable the tool's core value is (0–10).
        api_openness: Quality of public APIs/webhooks (0–10).
        workflow_complexity: Depth of embedding in workflows (0–10).
        data_sensitivity: Risk/friction of data migration (0–10).
        incumbent_inertia: Organizational switching cost (0–10).
        alternatives: Curated list of agentic/open-source alternatives.
        rationale: Brief explanation of the risk scores.
    """

    __slots__ = (
        "name",
        "category",
        "task_automation_ratio",
        "api_openness",
        "workflow_complexity",
        "data_sensitivity",
        "incumbent_inertia",
        "alternatives",
        "rationale",
    )

    def __init__(
        self,
        name: str,
        category: ToolCategory,
        task_automation_ratio: float,
        api_openness: float,
        workflow_complexity: float,
        data_sensitivity: float,
        incumbent_inertia: float,
        alternatives: list[str],
        rationale: str,
    ) -> None:
        """Initialize a KnowledgeBaseEntry with all required fields."""
        self.name = name
        self.category = category
        self.task_automation_ratio = task_automation_ratio
        self.api_openness = api_openness
        self.workflow_complexity = workflow_complexity
        self.data_sensitivity = data_sensitivity
        self.incumbent_inertia = incumbent_inertia
        self.alternatives = alternatives
        self.rationale = rationale

    def dimensions(self) -> dict[str, float]:
        """Return the five risk dimensions as a dictionary."""
        return {
            "task_automation_ratio": self.task_automation_ratio,
            "api_openness": self.api_openness,
            "workflow_complexity": self.workflow_complexity,
            "data_sensitivity": self.data_sensitivity,
            "incumbent_inertia": self.incumbent_inertia,
        }


# ---------------------------------------------------------------------------
# The static knowledge base — 50+ common SaaS tools
# ---------------------------------------------------------------------------

_RAW_ENTRIES: list[dict[str, object]] = [
    # --- Automation & Integration ---
    {
        "name": "Zapier",
        "category": ToolCategory.AUTOMATION,
        "task_automation_ratio": 9.5,
        "api_openness": 9.0,
        "workflow_complexity": 3.0,
        "data_sensitivity": 2.0,
        "incumbent_inertia": 3.5,
        "alternatives": ["n8n", "LangChain agents", "AutoGen", "Temporal.io", "Prefect"],
        "rationale": (
            "Zapier's entire value proposition is trigger-action automation—exactly "
            "what LLM agents do natively. Its open API and 6000+ integrations make "
            "it trivially replaceable by agentic frameworks. Low workflow complexity "
            "and data sensitivity accelerate displacement."
        ),
    },
    {
        "name": "Make",
        "category": ToolCategory.AUTOMATION,
        "task_automation_ratio": 9.0,
        "api_openness": 8.5,
        "workflow_complexity": 4.0,
        "data_sensitivity": 2.0,
        "incumbent_inertia": 3.0,
        "alternatives": ["n8n", "LangChain agents", "Activepieces", "Pipedream"],
        "rationale": (
            "Make (formerly Integromat) offers visual workflow automation with "
            "complex branching—still highly automatable by agents. Slightly more "
            "complex scenarios than Zapier but similar displacement trajectory."
        ),
    },
    {
        "name": "Workato",
        "category": ToolCategory.AUTOMATION,
        "task_automation_ratio": 8.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 5.0,
        "alternatives": ["n8n", "AutoGen", "LangChain", "Temporal.io"],
        "rationale": (
            "Enterprise automation platform with deeper integrations and more "
            "complex recipes than Zapier. Higher inertia due to enterprise contracts."
        ),
    },
    {
        "name": "Tray.io",
        "category": ToolCategory.AUTOMATION,
        "task_automation_ratio": 8.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 3.0,
        "incumbent_inertia": 4.0,
        "alternatives": ["n8n", "Prefect", "AutoGen", "LangChain"],
        "rationale": (
            "Developer-focused iPaaS platform. Strong API makes it replaceable "
            "by agent orchestration frameworks."
        ),
    },
    # --- CRM & Sales ---
    {
        "name": "Salesforce",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 8.5,
        "data_sensitivity": 8.0,
        "incumbent_inertia": 9.0,
        "alternatives": ["Twenty CRM", "Pipedrive + AI agents", "HubSpot CRM", "Attio"],
        "rationale": (
            "Salesforce has deep workflow complexity, decades of customer data, "
            "and extreme organizational inertia. While AI can automate data entry "
            "and forecasting, wholesale replacement is slow due to customization "
            "and compliance requirements."
        ),
    },
    {
        "name": "HubSpot",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 6.5,
        "api_openness": 8.0,
        "workflow_complexity": 6.0,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 6.0,
        "alternatives": ["Twenty CRM", "Attio", "Pipedrive", "AI-native CRM agents"],
        "rationale": (
            "HubSpot combines CRM, marketing, and support. Moderate complexity "
            "and good API openness make it partially displaceable, especially the "
            "marketing automation components."
        ),
    },
    {
        "name": "Pipedrive",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 6.0,
        "api_openness": 8.0,
        "workflow_complexity": 4.5,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 4.5,
        "alternatives": ["Twenty CRM", "Attio", "Folk CRM", "custom LLM pipeline"],
        "rationale": (
            "Sales-focused CRM with good API. Simpler than Salesforce and more "
            "amenable to agent-driven pipeline management."
        ),
    },
    {
        "name": "Attio",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 7.0,
        "api_openness": 9.0,
        "workflow_complexity": 4.0,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 3.0,
        "alternatives": ["Twenty CRM", "custom LangChain agent CRM"],
        "rationale": (
            "Modern API-first CRM. High openness and low inertia make it "
            "both easy to integrate with agents and easy to replace."
        ),
    },
    # --- Customer Support ---
    {
        "name": "Intercom",
        "category": ToolCategory.CUSTOMER_SUPPORT,
        "task_automation_ratio": 8.0,
        "api_openness": 8.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.5,
        "alternatives": [
            "OpenAgents", "custom LLM chatbot (LangChain)", "Chatwoot", "Botpress"
        ],
        "rationale": (
            "In-app chat and support ticketing is a primary use case for LLM agents. "
            "Intercom's own Fin AI shows the category is self-disrupting. "
            "High automation ratio; moderate inertia from workflow integrations."
        ),
    },
    {
        "name": "Zendesk",
        "category": ToolCategory.CUSTOMER_SUPPORT,
        "task_automation_ratio": 7.5,
        "api_openness": 8.5,
        "workflow_complexity": 6.0,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 6.5,
        "alternatives": ["Freshdesk", "Chatwoot", "custom LLM support agent", "Botpress"],
        "rationale": (
            "Ticket management and macro-based automation is highly amenable to "
            "LLM agents. Large enterprises face higher switching costs due to "
            "custom integrations and historical ticket data."
        ),
    },
    {
        "name": "Freshdesk",
        "category": ToolCategory.CUSTOMER_SUPPORT,
        "task_automation_ratio": 7.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.0,
        "alternatives": ["Chatwoot", "Zammad", "custom LLM support agent"],
        "rationale": (
            "Similar to Zendesk but typically deployed at smaller scale. "
            "Lower inertia makes it more quickly displaceable."
        ),
    },
    {
        "name": "Drift",
        "category": ToolCategory.CUSTOMER_SUPPORT,
        "task_automation_ratio": 9.0,
        "api_openness": 7.5,
        "workflow_complexity": 4.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 4.0,
        "alternatives": ["custom LLM chatbot", "Botpress", "OpenAgents", "Chatwoot"],
        "rationale": (
            "Conversational marketing chatbot—a task LLMs perform natively. "
            "Very high displacement risk with minimal workflow complexity."
        ),
    },
    # --- Knowledge Management ---
    {
        "name": "Notion",
        "category": ToolCategory.KNOWLEDGE_MANAGEMENT,
        "task_automation_ratio": 5.0,
        "api_openness": 7.0,
        "workflow_complexity": 6.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 6.0,
        "alternatives": ["Obsidian + AI plugins", "Mem.ai", "Outline", "AppFlowy"],
        "rationale": (
            "Notion combines docs, databases, and project tracking. AI can auto-generate "
            "and organize content, but deep embedding in team workflows adds inertia. "
            "The API enables good agent integration."
        ),
    },
    {
        "name": "Confluence",
        "category": ToolCategory.KNOWLEDGE_MANAGEMENT,
        "task_automation_ratio": 4.5,
        "api_openness": 7.0,
        "workflow_complexity": 6.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 7.0,
        "alternatives": ["Outline", "Notion", "GitBook", "Docusaurus + AI"],
        "rationale": (
            "Enterprise wiki deeply embedded in engineering workflows. High inertia "
            "from Atlassian ecosystem lock-in. AI can help with content generation "
            "but full replacement requires significant migration effort."
        ),
    },
    {
        "name": "Guru",
        "category": ToolCategory.KNOWLEDGE_MANAGEMENT,
        "task_automation_ratio": 6.5,
        "api_openness": 6.5,
        "workflow_complexity": 4.5,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 4.5,
        "alternatives": ["Notion AI", "Outline", "custom RAG pipeline"],
        "rationale": (
            "Knowledge base tool focused on verified cards. RAG-based agent pipelines "
            "can replicate its core value proposition effectively."
        ),
    },
    {
        "name": "Coda",
        "category": ToolCategory.KNOWLEDGE_MANAGEMENT,
        "task_automation_ratio": 5.5,
        "api_openness": 7.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 4.5,
        "alternatives": ["Notion", "AppFlowy", "custom agent dashboards"],
        "rationale": (
            "Doc-meets-spreadsheet tool. Moderate displacement risk; its programmability "
            "makes it a good integration target for agents."
        ),
    },
    # --- Project Management ---
    {
        "name": "Jira",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 5.0,
        "api_openness": 8.0,
        "workflow_complexity": 8.0,
        "data_sensitivity": 4.5,
        "incumbent_inertia": 8.5,
        "alternatives": ["Linear", "Plane", "Height", "GitHub Issues + AI"],
        "rationale": (
            "Jira is deeply embedded in engineering workflows with heavily customized "
            "schemes. Very high inertia from Atlassian ecosystem and years of project history. "
            "AI can assist but full replacement is slow."
        ),
    },
    {
        "name": "Linear",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 5.5,
        "api_openness": 9.0,
        "workflow_complexity": 4.5,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 4.0,
        "alternatives": ["Plane", "GitHub Issues + AI", "Height", "custom agent tracker"],
        "rationale": (
            "Modern, API-first project tracker with clean data model. Lower inertia "
            "than Jira; excellent API enables deep agent integration for triage and "
            "sprint planning automation."
        ),
    },
    {
        "name": "Asana",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 5.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 5.5,
        "alternatives": ["Plane", "Linear", "ClickUp", "Notion"],
        "rationale": (
            "Work management platform used across teams. Good API and moderate complexity "
            "make it displaceable over the mid-term horizon."
        ),
    },
    {
        "name": "Monday.com",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 5.0,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 5.0,
        "alternatives": ["Plane", "Linear", "ClickUp", "Notion"],
        "rationale": (
            "Visual work management tool. Its automation features are being commoditized "
            "by AI agents. Moderate displacement risk."
        ),
    },
    {
        "name": "ClickUp",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 6.0,
        "api_openness": 8.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 4.5,
        "alternatives": ["Plane", "Linear", "Notion"],
        "rationale": (
            "All-in-one work platform. Broad feature set increases complexity "
            "but also increases surface area for agent augmentation."
        ),
    },
    # --- Communication ---
    {
        "name": "Slack",
        "category": ToolCategory.COMMUNICATION,
        "task_automation_ratio": 4.0,
        "api_openness": 8.5,
        "workflow_complexity": 7.5,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 8.0,
        "alternatives": ["Mattermost", "Discord", "Matrix/Element", "Zulip"],
        "rationale": (
            "Slack is the communication backbone for most teams—very high inertia "
            "and workflow complexity. Excellent API enables agent integration, but "
            "replacement rather than augmentation is unlikely near-term."
        ),
    },
    {
        "name": "Microsoft Teams",
        "category": ToolCategory.COMMUNICATION,
        "task_automation_ratio": 3.5,
        "api_openness": 7.0,
        "workflow_complexity": 8.0,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 9.0,
        "alternatives": ["Mattermost", "Slack", "Matrix/Element"],
        "rationale": (
            "Deep Microsoft 365 integration creates very high inertia. Replacement "
            "is unlikely given enterprise contracts and Active Directory dependencies."
        ),
    },
    {
        "name": "Zoom",
        "category": ToolCategory.COMMUNICATION,
        "task_automation_ratio": 4.0,
        "api_openness": 7.5,
        "workflow_complexity": 5.0,
        "data_sensitivity": 4.5,
        "incumbent_inertia": 6.0,
        "alternatives": ["Jitsi", "Google Meet", "Around", "AI meeting assistants"],
        "rationale": (
            "Video conferencing with growing AI features (transcription, summaries). "
            "Core meeting functionality is hard to replace but AI assistants are "
            "layered on top rather than displacing Zoom itself."
        ),
    },
    # --- HR & People Ops ---
    {
        "name": "Workday",
        "category": ToolCategory.HR,
        "task_automation_ratio": 4.0,
        "api_openness": 5.5,
        "workflow_complexity": 8.5,
        "data_sensitivity": 9.5,
        "incumbent_inertia": 9.0,
        "alternatives": ["Rippling", "BambooHR + AI workflows", "HiBob"],
        "rationale": (
            "Enterprise HRIS with payroll, compliance, and benefits. Extremely high "
            "data sensitivity (PII, payroll) and incumbent inertia. One of the lowest "
            "displacement risk tools in any stack."
        ),
    },
    {
        "name": "BambooHR",
        "category": ToolCategory.HR,
        "task_automation_ratio": 5.0,
        "api_openness": 7.0,
        "workflow_complexity": 6.0,
        "data_sensitivity": 8.5,
        "incumbent_inertia": 6.5,
        "alternatives": ["Rippling", "HiBob", "Gusto"],
        "rationale": (
            "SMB-focused HRIS. Better API than Workday but still high data sensitivity "
            "from employee records. Mid-term displacement possible for smaller orgs."
        ),
    },
    {
        "name": "Greenhouse",
        "category": ToolCategory.HR,
        "task_automation_ratio": 6.5,
        "api_openness": 8.0,
        "workflow_complexity": 6.0,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 5.5,
        "alternatives": [
            "Lever", "Ashby", "custom LLM recruiting pipeline", "AI screening agents"
        ],
        "rationale": (
            "ATS with growing AI screening features. Resume parsing and candidate "
            "ranking are prime agent territory. Good API enables integration with "
            "AI-native recruiting tools."
        ),
    },
    {
        "name": "Rippling",
        "category": ToolCategory.HR,
        "task_automation_ratio": 5.5,
        "api_openness": 7.0,
        "workflow_complexity": 7.0,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 7.0,
        "alternatives": ["Deel", "Gusto", "BambooHR"],
        "rationale": (
            "Modern HR/IT platform with strong automation. High data sensitivity from "
            "payroll and device management reduces near-term displacement risk."
        ),
    },
    {
        "name": "Lattice",
        "category": ToolCategory.HR,
        "task_automation_ratio": 6.0,
        "api_openness": 6.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 5.0,
        "alternatives": ["Culture Amp", "15Five", "custom agent performance reviews"],
        "rationale": (
            "Performance management and engagement platform. AI can generate review "
            "summaries and goal suggestions, making portions highly displaceable."
        ),
    },
    # --- Finance ---
    {
        "name": "QuickBooks",
        "category": ToolCategory.FINANCE,
        "task_automation_ratio": 5.5,
        "api_openness": 6.5,
        "workflow_complexity": 6.0,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 7.0,
        "alternatives": ["Wave", "Xero", "Sage Intacct", "AI bookkeeping agents"],
        "rationale": (
            "Accounting software with high financial data sensitivity and regulatory "
            "requirements. AI can automate categorization but full replacement requires "
            "significant compliance review."
        ),
    },
    {
        "name": "Xero",
        "category": ToolCategory.FINANCE,
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 6.0,
        "alternatives": ["Wave", "QuickBooks", "custom AI bookkeeping"],
        "rationale": (
            "Cloud accounting with better API than QuickBooks. Still high sensitivity "
            "from financial data and audit requirements."
        ),
    },
    {
        "name": "Stripe",
        "category": ToolCategory.FINANCE,
        "task_automation_ratio": 7.0,
        "api_openness": 10.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 9.0,
        "incumbent_inertia": 7.0,
        "alternatives": ["Paddle", "Lemon Squeezy", "Lago (open-source billing)"],
        "rationale": (
            "Payment infrastructure with best-in-class API. Very high data sensitivity "
            "(PCI-DSS) and switching costs from payment method storage limit displacement. "
            "AI augments Stripe rather than replacing it."
        ),
    },
    {
        "name": "Brex",
        "category": ToolCategory.FINANCE,
        "task_automation_ratio": 6.0,
        "api_openness": 7.5,
        "workflow_complexity": 4.5,
        "data_sensitivity": 8.5,
        "incumbent_inertia": 5.0,
        "alternatives": ["Ramp", "Mercury", "custom expense automation"],
        "rationale": (
            "Corporate cards and expense management. AI can automate receipt matching "
            "and policy enforcement. Financial data sensitivity is the main brake."
        ),
    },
    # --- Data Analytics ---
    {
        "name": "Looker",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 5.5,
        "api_openness": 7.5,
        "workflow_complexity": 7.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 7.0,
        "alternatives": ["Metabase", "Apache Superset", "Evidence.dev", "AI SQL agents"],
        "rationale": (
            "Enterprise BI with LookML semantic layer. AI can generate SQL and explain "
            "dashboards but full LookML model replacement is complex. High inertia from "
            "business-team dependency."
        ),
    },
    {
        "name": "Tableau",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 4.5,
        "api_openness": 6.5,
        "workflow_complexity": 7.0,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 7.5,
        "alternatives": ["Apache Superset", "Metabase", "Evidence.dev", "Observable"],
        "rationale": (
            "Tableau's drag-and-drop interface and embedded analytics have high user "
            "inertia. AI assistants can generate charts, but enterprise deployments "
            "with custom data sources are slow to migrate."
        ),
    },
    {
        "name": "Segment",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 7.0,
        "api_openness": 9.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 6.0,
        "alternatives": ["RudderStack", "Jitsu", "PostHog", "custom event pipeline"],
        "rationale": (
            "Customer data platform with routing logic that agents can replicate. "
            "Excellent API but high data sensitivity from PII event streams. "
            "Open-source alternatives (RudderStack) are mature."
        ),
    },
    {
        "name": "Mixpanel",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 5.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 5.5,
        "alternatives": ["PostHog", "Amplitude", "custom analytics stack"],
        "rationale": (
            "Product analytics with funnel and retention analysis. AI can generate "
            "insights from raw event data. PostHog offers open-source parity."
        ),
    },
    {
        "name": "Amplitude",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 5.5,
        "api_openness": 8.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 5.5,
        "alternatives": ["PostHog", "Mixpanel", "custom analytics + AI"],
        "rationale": (
            "Similar to Mixpanel with stronger behavioral cohort features. "
            "Mid-term displacement as AI-native analytics tools mature."
        ),
    },
    {
        "name": "dbt Cloud",
        "category": ToolCategory.DATA_ANALYTICS,
        "task_automation_ratio": 6.0,
        "api_openness": 8.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 5.0,
        "alternatives": ["dbt Core (self-hosted)", "SQLMesh", "AI SQL generation"],
        "rationale": (
            "Data transformation platform. The open-source dbt Core is a direct "
            "alternative. AI-assisted SQL generation reduces the value-add of managed "
            "cloud features."
        ),
    },
    # --- Marketing ---
    {
        "name": "Marketo",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 7.5,
        "api_openness": 7.5,
        "workflow_complexity": 7.0,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 7.5,
        "alternatives": ["Mautic", "Customer.io", "ActiveCampaign", "LangChain email agents"],
        "rationale": (
            "Enterprise marketing automation with complex nurture programs. High inertia "
            "from Salesforce integration and campaign history. AI can replace individual "
            "campaigns but full platform migration is slow."
        ),
    },
    {
        "name": "Mailchimp",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 7.0,
        "api_openness": 8.0,
        "workflow_complexity": 4.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.0,
        "alternatives": ["Listmonk", "Mautic", "Customer.io", "SendGrid + AI"],
        "rationale": (
            "Email marketing platform at SMB scale. AI can generate copy and optimize "
            "send times. Open-source alternatives like Listmonk are viable replacements."
        ),
    },
    {
        "name": "Customer.io",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 7.5,
        "api_openness": 9.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 5.0,
        "alternatives": ["Mautic", "Braze", "custom LLM messaging agent"],
        "rationale": (
            "Developer-friendly messaging platform with excellent API. High automation "
            "ratio as behavioral triggers are prime agent territory."
        ),
    },
    {
        "name": "Braze",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 7.0,
        "api_openness": 8.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 6.0,
        "incumbent_inertia": 6.5,
        "alternatives": ["Customer.io", "Mautic", "custom personalization agent"],
        "rationale": (
            "Enterprise customer engagement platform. Strong AI features already built-in. "
            "Higher inertia from mobile push/in-app capabilities."
        ),
    },
    # --- DevTools ---
    {
        "name": "GitHub",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 5.0,
        "api_openness": 9.5,
        "workflow_complexity": 8.0,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 8.5,
        "alternatives": ["GitLab (self-hosted)", "Gitea", "Sourcehut"],
        "rationale": (
            "Core development infrastructure—very high workflow complexity and inertia. "
            "AI Copilot augments GitHub rather than displacing it. Git hosting has "
            "high switching cost from CI/CD integrations and automation."
        ),
    },
    {
        "name": "GitLab",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 5.0,
        "api_openness": 9.0,
        "workflow_complexity": 8.0,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 7.5,
        "alternatives": ["GitHub", "Gitea", "Sourcehut"],
        "rationale": (
            "All-in-one DevOps platform. Self-hosted option reduces some lock-in. "
            "Similar displacement profile to GitHub."
        ),
    },
    {
        "name": "Datadog",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 5.5,
        "api_openness": 8.5,
        "workflow_complexity": 7.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 7.0,
        "alternatives": [
            "Grafana + Prometheus", "OpenTelemetry stack", "SigNoz", "Hyperdx"
        ],
        "rationale": (
            "Observability platform with deep agent integrations. AI can help with "
            "anomaly detection and alerting, but the data ingestion pipeline creates "
            "significant switching friction. Open-source stacks are viable."
        ),
    },
    {
        "name": "PagerDuty",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 6.5,
        "api_openness": 9.0,
        "workflow_complexity": 6.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 6.5,
        "alternatives": ["Grafana OnCall", "OpsGenie", "incident.io", "custom alert agent"],
        "rationale": (
            "Incident management platform. AI agents can triage and route alerts, "
            "making this a medium-term displacement target. Good API supports integration."
        ),
    },
    {
        "name": "Sentry",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 6.0,
        "api_openness": 8.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 5.0,
        "incumbent_inertia": 5.0,
        "alternatives": ["GlitchTip", "Rollbar", "Bugsink", "custom error tracking"],
        "rationale": (
            "Error monitoring with AI-assisted root cause analysis already built in. "
            "Open-source alternative GlitchTip is compatible. Mid-term displacement risk."
        ),
    },
    {
        "name": "Vercel",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 5.0,
        "api_openness": 8.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 5.5,
        "alternatives": ["Netlify", "Coolify", "Railway", "self-hosted infra"],
        "rationale": (
            "Frontend deployment platform. AI can scaffold deployment configs but the "
            "DX and edge network are valuable. Open-source alternatives exist."
        ),
    },
    # --- Security ---
    {
        "name": "Okta",
        "category": ToolCategory.SECURITY,
        "task_automation_ratio": 4.0,
        "api_openness": 8.0,
        "workflow_complexity": 7.5,
        "data_sensitivity": 9.5,
        "incumbent_inertia": 9.0,
        "alternatives": ["Keycloak", "Authentik", "Zitadel"],
        "rationale": (
            "Identity provider deeply embedded in every application's auth flow. "
            "Extremely high data sensitivity and inertia. Replacement risk is very low "
            "despite good API availability."
        ),
    },
    {
        "name": "1Password",
        "category": ToolCategory.SECURITY,
        "task_automation_ratio": 3.0,
        "api_openness": 7.0,
        "workflow_complexity": 5.0,
        "data_sensitivity": 9.5,
        "incumbent_inertia": 7.0,
        "alternatives": ["Bitwarden", "Vaultwarden (self-hosted)", "KeePassXC"],
        "rationale": (
            "Password manager with very high data sensitivity. Switching risk from "
            "stored credentials makes displacement unlikely. Bitwarden is a mature "
            "open-source alternative if migration is planned."
        ),
    },
    # --- Storage ---
    {
        "name": "Dropbox",
        "category": ToolCategory.STORAGE,
        "task_automation_ratio": 3.5,
        "api_openness": 8.0,
        "workflow_complexity": 4.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 5.5,
        "alternatives": ["Nextcloud", "S3 + rclone", "Seafile"],
        "rationale": (
            "File sync and storage. AI adds search and organization but core sync "
            "functionality is a commodity. Open-source self-hosted alternatives are mature."
        ),
    },
    {
        "name": "Box",
        "category": ToolCategory.STORAGE,
        "task_automation_ratio": 4.0,
        "api_openness": 8.5,
        "workflow_complexity": 5.5,
        "data_sensitivity": 7.5,
        "incumbent_inertia": 6.5,
        "alternatives": ["Nextcloud", "SharePoint", "Egnyte"],
        "rationale": (
            "Enterprise content management with compliance features. Higher inertia "
            "from regulatory workflow integrations (e-signature, legal hold)."
        ),
    },
    # --- eCommerce ---
    {
        "name": "Shopify",
        "category": ToolCategory.ECOMMERCE,
        "task_automation_ratio": 5.5,
        "api_openness": 8.5,
        "workflow_complexity": 6.5,
        "data_sensitivity": 7.0,
        "incumbent_inertia": 7.0,
        "alternatives": ["WooCommerce", "Medusa.js", "Saleor", "custom storefront"],
        "rationale": (
            "eCommerce platform with excellent API (Storefront + Admin). AI can generate "
            "product descriptions and personalize recommendations. Full replacement "
            "requires migrating product catalog and payment methods."
        ),
    },
    # --- Additional common tools ---
    {
        "name": "Airtable",
        "category": ToolCategory.PROJECT_MANAGEMENT,
        "task_automation_ratio": 6.5,
        "api_openness": 8.5,
        "workflow_complexity": 5.0,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 5.0,
        "alternatives": ["NocoDB", "Baserow", "Grist", "custom agent database"],
        "rationale": (
            "Flexible database-spreadsheet hybrid. AI can build similar structured "
            "data pipelines. NocoDB is a solid open-source alternative."
        ),
    },
    {
        "name": "Figma",
        "category": ToolCategory.DEVTOOLS,
        "task_automation_ratio": 4.0,
        "api_openness": 8.0,
        "workflow_complexity": 6.5,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 7.5,
        "alternatives": ["Penpot", "Framer", "AI design generation tools"],
        "rationale": (
            "Design tool with strong collaborative features. AI generates designs but "
            "Figma remains the collaboration layer. High inertia from design system assets."
        ),
    },
    {
        "name": "Loom",
        "category": ToolCategory.COMMUNICATION,
        "task_automation_ratio": 6.0,
        "api_openness": 7.0,
        "workflow_complexity": 3.5,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 4.0,
        "alternatives": ["Tella", "Cap", "Clipchamp + AI transcription"],
        "rationale": (
            "Async video messaging. AI transcription and summarization commoditize "
            "Loom's core value. Low complexity and inertia accelerate displacement."
        ),
    },
    {
        "name": "Calendly",
        "category": ToolCategory.AUTOMATION,
        "task_automation_ratio": 8.5,
        "api_openness": 7.5,
        "workflow_complexity": 3.0,
        "data_sensitivity": 3.0,
        "incumbent_inertia": 3.5,
        "alternatives": ["Cal.com", "Acuity", "AI scheduling agents"],
        "rationale": (
            "Scheduling automation is trivially handled by AI agents. Cal.com is a "
            "mature open-source drop-in. Very high displacement risk."
        ),
    },
    {
        "name": "Typeform",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 7.5,
        "api_openness": 8.0,
        "workflow_complexity": 3.5,
        "data_sensitivity": 4.0,
        "incumbent_inertia": 3.5,
        "alternatives": ["Tally", "Formbricks", "custom conversational AI form"],
        "rationale": (
            "Conversational forms are easily replicated by LLM agents. Tally and "
            "Formbricks offer free/open-source alternatives with equivalent UX."
        ),
    },
    {
        "name": "Webflow",
        "category": ToolCategory.MARKETING,
        "task_automation_ratio": 5.0,
        "api_openness": 7.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 3.5,
        "incumbent_inertia": 5.5,
        "alternatives": ["Framer", "WordPress + AI", "custom Next.js site"],
        "rationale": (
            "Visual web development tool. AI code generation reduces the value "
            "proposition but existing sites have high migration friction."
        ),
    },
    {
        "name": "Apollo.io",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 8.0,
        "api_openness": 8.0,
        "workflow_complexity": 4.5,
        "data_sensitivity": 5.5,
        "incumbent_inertia": 4.5,
        "alternatives": ["Clay", "custom LLM prospecting agent", "PhantomBuster"],
        "rationale": (
            "Sales intelligence and outreach platform. AI agents can replicate "
            "prospecting workflows. High automation ratio drives near-term displacement."
        ),
    },
    {
        "name": "Gong",
        "category": ToolCategory.CRM,
        "task_automation_ratio": 7.0,
        "api_openness": 7.0,
        "workflow_complexity": 5.5,
        "data_sensitivity": 6.5,
        "incumbent_inertia": 6.0,
        "alternatives": ["Fathom", "Fireflies.ai", "custom call analysis agent"],
        "rationale": (
            "Revenue intelligence via call recording/analysis. This is a native LLM "
            "use case; many open alternatives exist. Mid-term displacement likely."
        ),
    },
]


# ---------------------------------------------------------------------------
# Build lookup index from raw entries
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, KnowledgeBaseEntry] = {}

for _raw in _RAW_ENTRIES:
    _entry = KnowledgeBaseEntry(
        name=str(_raw["name"]),
        category=_raw["category"],  # type: ignore[arg-type]
        task_automation_ratio=float(_raw["task_automation_ratio"]),  # type: ignore[arg-type]
        api_openness=float(_raw["api_openness"]),  # type: ignore[arg-type]
        workflow_complexity=float(_raw["workflow_complexity"]),  # type: ignore[arg-type]
        data_sensitivity=float(_raw["data_sensitivity"]),  # type: ignore[arg-type]
        incumbent_inertia=float(_raw["incumbent_inertia"]),  # type: ignore[arg-type]
        alternatives=list(_raw["alternatives"]),  # type: ignore[arg-type]
        rationale=str(_raw["rationale"]),
    )
    # Index by lowercase name for case-insensitive lookup
    _KNOWLEDGE_BASE[_entry.name.lower()] = _entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup(tool_name: str) -> Optional[KnowledgeBaseEntry]:
    """Look up a SaaS tool in the knowledge base by name (case-insensitive).

    Args:
        tool_name: The name of the SaaS tool to look up.

    Returns:
        A KnowledgeBaseEntry if the tool is found, or None if unknown.
    """
    return _KNOWLEDGE_BASE.get(tool_name.lower().strip())


def is_known(tool_name: str) -> bool:
    """Return True if the tool is in the knowledge base (case-insensitive).

    Args:
        tool_name: The name of the SaaS tool to check.

    Returns:
        True if the tool is known, False otherwise.
    """
    return tool_name.lower().strip() in _KNOWLEDGE_BASE


def all_entries() -> list[KnowledgeBaseEntry]:
    """Return all knowledge base entries as a list.

    Returns:
        List of all KnowledgeBaseEntry instances.
    """
    return list(_KNOWLEDGE_BASE.values())


def known_tool_names() -> list[str]:
    """Return a sorted list of all known tool names (in their canonical display form).

    Returns:
        Sorted list of tool name strings.
    """
    return sorted(entry.name for entry in _KNOWLEDGE_BASE.values())


def entries_by_category(category: ToolCategory) -> list[KnowledgeBaseEntry]:
    """Return all knowledge base entries matching the given category.

    Args:
        category: The ToolCategory to filter by.

    Returns:
        List of KnowledgeBaseEntry instances in the given category.
    """
    return [entry for entry in _KNOWLEDGE_BASE.values() if entry.category == category]


def knowledge_base_size() -> int:
    """Return the total number of tools in the knowledge base.

    Returns:
        Integer count of knowledge base entries.
    """
    return len(_KNOWLEDGE_BASE)
