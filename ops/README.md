# DataForge operations kit

Runbooks, backup/restore scripts, load tests, and SLO targets that sit beside the application code.

## Owner checklist (prove production readiness)

Tick these after deploying; they cannot be completed from source alone.

| Step | Where | Done |
|---|---|---|
| Restore a non-prod Postgres dump and record the date | [backup/RESTORE.md](backup/RESTORE.md) | |
| Point uptime / Render health checks at `/api/v1/ready` | [runbooks/degraded-health.md](runbooks/degraded-health.md) | |
| Scrape `/api/v1/metrics` or enable Sentry alerts | [runbooks/degraded-health.md](runbooks/degraded-health.md) | |
| Run k6 smoke against staging and paste the summary | [load/README.md](load/README.md) | |

## Contents

- [slo.md](slo.md) — latency, availability, RPO/RTO targets
- [runbooks/session-auth.md](runbooks/session-auth.md) — rate limits, MFA, session revoke
- [runbooks/degraded-health.md](runbooks/degraded-health.md) — `/health`, `/ready`, metrics
- [backup/](backup/) — `pg_dump` / `pg_restore` helpers
- [load/](load/) — k6 smoke script

## Related app docs

- Backend: `dataforge.backend/README.md`
- Frontend: `skills-quest-forge/README.md`
