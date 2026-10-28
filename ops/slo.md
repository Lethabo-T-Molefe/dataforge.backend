# Service level objectives (defaults)

Update these when product requirements change. They are targets, not guarantees, until monitored in production.

## Availability

| Service | Target | Measurement |
|---|---|---|
| API readiness (`GET /api/v1/ready`) | 99.5% monthly | Uptime checks; 503 counts as down |
| Web app static origin | 99.9% monthly | CDN / Render static health |

Partial degradation (e.g. Redis down while DB is up) should surface as `/health` `status: degraded` and `/ready` 503 when Redis is configured.

## Latency (API, p95 over 5 minutes)

| Journey | Target |
|---|---|
| `GET /api/v1/health` | &lt; 200 ms |
| `POST /api/v1/auth/login` | &lt; 800 ms |
| Dataset list / workspace bootstrap | &lt; 1.5 s |
| Upload complete (excluding transfer) | &lt; 30 s for files under 50 MB |
| Ask assistant turn | &lt; 15 s when LLM is enabled |

## Frontend Core Web Vitals (75th percentile)

| Metric | Good |
|---|---|
| LCP | ≤ 2.5 s |
| INP | ≤ 200 ms |
| CLS | ≤ 0.1 |

Instrument via `web-vitals` in the SPA (`[web-vitals]` console / optional `VITE_VITALS_ENDPOINT`).

## Recovery

| Objective | Default |
|---|---|
| RPO (max acceptable data loss) | 24 hours (daily Postgres dump) |
| RTO (max time to restore service) | 4 hours for API + database |

File storage (local or Cloudinary) and encryption keys must be included in the backup plan — see [backup/RESTORE.md](backup/RESTORE.md).
