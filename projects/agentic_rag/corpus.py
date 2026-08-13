"""Original, fictional 'Aurora Cloud' API documentation used as the demo
knowledge base for the agentic RAG project."""

SEED_DOCUMENTS = [
    {
        "id": "auth",
        "title": "Authentication & API Keys",
        "text": (
            "Aurora Cloud authenticates requests with API keys or OAuth2 "
            "client-credentials tokens. API keys should be rotated at least "
            "every 90 days, and each key can be scoped to read-only or "
            "read-write access. Keys are shown once at creation time and "
            "cannot be retrieved again, only revoked and replaced."
        ),
    },
    {
        "id": "rate-limits",
        "title": "Rate Limits",
        "text": (
            "The standard tier allows 600 requests per minute per API key. "
            "The enterprise tier raises this to 5000 requests per minute. "
            "Requests beyond the limit receive a 429 response with a "
            "Retry-After header indicating how many seconds to wait before "
            "retrying."
        ),
    },
    {
        "id": "billing",
        "title": "Billing & Pricing Tiers",
        "text": (
            "The Free tier includes 10,000 requests per month at no cost. "
            "The Pro tier costs $49 per month and includes 500,000 "
            "requests, with usage-based overage billing beyond that. "
            "Enterprise pricing is custom and negotiated with dedicated "
            "account management."
        ),
    },
    {
        "id": "webhooks",
        "title": "Webhooks",
        "text": (
            "Webhook payloads are signed with HMAC-SHA256; verify the "
            "X-Aurora-Signature header against your webhook secret before "
            "trusting a payload. Failed deliveries are retried with "
            "exponential backoff for up to 5 attempts before the event is "
            "marked as dead-lettered."
        ),
    },
    {
        "id": "security",
        "title": "Security & Compliance",
        "text": (
            "Aurora Cloud is SOC 2 Type II certified. Data is encrypted at "
            "rest using AES-256 and in transit using TLS 1.3. An "
            "independent penetration test is conducted annually, and "
            "results are available to enterprise customers under NDA."
        ),
    },
    {
        "id": "data-retention",
        "title": "Data Retention Policy",
        "text": (
            "Request logs are retained for 30 days for debugging purposes. "
            "Database backups are retained for 90 days. When an account is "
            "deleted, all associated data is purged from primary storage "
            "and backups within 30 days."
        ),
    },
    {
        "id": "support-sla",
        "title": "Support SLAs",
        "text": (
            "Free tier customers receive community forum support only. Pro "
            "tier customers get email support with a 24-hour response "
            "time. Enterprise customers get 24/7 support with a 1-hour "
            "response time and a dedicated shared Slack channel."
        ),
    },
    {
        "id": "gdpr",
        "title": "GDPR & Data Residency",
        "text": (
            "Customers in the European Union may opt into EU-only data "
            "residency. A Data Processing Agreement (DPA) is available on "
            "request. Data subjects can exercise the right to erasure "
            "through the /v1/privacy/erase endpoint or by contacting "
            "support."
        ),
    },
]
