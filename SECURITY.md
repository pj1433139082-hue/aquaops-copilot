# Security policy

## Scope

This policy covers the public AquaOps portfolio candidate and its reproducible demo. The candidate
uses only registered public sources and synthetic fixtures. It is decision-support software, not a
production control plane, and it has no security certification or availability guarantee.

The public surface intentionally has no write, delete, device-control, private-data or arbitrary
URL tools. Database and service credentials are required through local environment variables and
must never be committed or pasted into an issue, log or demo transcript.

## Reporting a vulnerability

Do not disclose secrets, private plant data, internal URLs or an exploitable proof of concept in a
public issue or pull request. Before external publication, the project owner must enable a private
security contact (for example, a repository security advisory channel). Until that channel exists,
keep the report local and share only with the project owner through an approved private channel.

Include the affected public file or version, a minimal reproducible description using synthetic
data, impact, and a safe mitigation. Redact credentials and proprietary operational details.

## Response expectations

The maintainer will acknowledge a report, reproduce it against the public staging tree, classify
severity, and publish a fix or mitigation only after the reporter confirms that no private material
is exposed. If a credential may have leaked, rotate it immediately and invalidate affected tokens;
do not wait for a code fix.

## Known limitations

The public demo does not model every production threat, tenant boundary or operational integration.
Do not connect it to a production Qdrant/PostgreSQL/Redis instance, real water-plant data or a
control system. External publication, image release and production deployment remain separate
approval gates.
