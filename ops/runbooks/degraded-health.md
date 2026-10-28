# Runbook: health, readiness, and metrics

## Endpoints

| Path | Behaviour |
|---|---|
| `GET /api/v1/health` | Always HTTP 200. Body `status` is `ok` or `degraded`. Includes `database`, `sandbox`, optional `redis`. |
| `GET /api/v1/ready` | HTTP 200 when ready; **503** when DB (or configured Redis) is unreachable or sandbox role missing. Use this for load-balancer / Render health checks. |
| `GET /api/v1/metrics` | Prometheus text exposition (request counts, latency histogram, 5xx counter). |

## Alerting suggestions

1. Probe `/api/v1/ready` every 60s; alert after 2 consecutive failures.
2. Scrape `/api/v1/metrics` every 15–60s; alert on rising `dataforge_http_5xx_total` rate.
3. If `SENTRY_DSN` is set, alert on new unresolved issues in production.

## Common causes of 503 on `/ready`

- Postgres down or migrations not applied (`sandbox` missing role).
- `REDIS_URL` set but Redis unreachable (rate-limit store).

## Temporary mitigation

- If Redis is optional for your deployment, unset `REDIS_URL` so rate limits fall back to in-memory (per replica) and readiness no longer depends on Redis.
- Restore Postgres from the latest dump ([../backup/RESTORE.md](../backup/RESTORE.md)).
