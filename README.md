# DataForge backend

The API behind DataForge: governed analytics for everyday members and data scientists. It covers identity
with MFA, roles and quotas, uploads and read-only connectors, PII detection and masking, quality profiling,
guided cleaning recipes, the bronze/silver/gold medallion warehouse, a semantic layer of shared metrics,
point-and-click analysis, an AI assistant that answers or refuses, SQL and notebooks in a sandbox, dashboards
and embedding, monitors with root cause, explainable predictions with PDF and Excel export, an immutable
audit trail, search, and Slack.

Apache Superset (dashboards, Explore, SQL Lab) and JupyterLab (notebooks) run beside it. Both are themed
as DataForge and governed by it (see [Superset and Jupyter](#superset-and-jupyter)).

## Architecture

FastAPI (async) in a Clean Architecture layout:

```
src/dataforge_backend/
  core/            settings, database engine, security (argon2id, JWT, TOTP), encryption, errors, logging
  domain/          framework-free rules: values, PII, profiling, insights, cleaning operations and recipes,
                   analysis engine and SQL, SQL guard, semantic layer, monitoring, prediction, assistant planner
  application/     use cases: identity, datasets, cleaning, analysis, workbench, assistant, metrics,
                   dashboards, monitors, predictions, connectors, admin, search, integrations, Slack, audit
  infrastructure/  PostgreSQL models, warehouse (COPY), sandbox, storage, email, Claude, Superset, Jupyter,
                   Slack, external databases
  presentation/    HTTP: v1 routers, request/response shapes (camelCase), auth dependencies, errors
  seed.py, cli.py  demo data and operational commands
integrations/
  superset/        Superset 6.1 config, DataForge theme, single sign-on, bootstrap (own virtual environment)
  jupyter/         JupyterLab 4.6 config, theme, single sign-on, per-member access, `dataforge` client library
```

PostgreSQL holds the metadata in schema `app` and member data in schemas `bronze` (raw text, immutable per
version), `silver` (cleaned, typed, masked) and `gold` (analytics-ready).

### Governance, built in

- **Journey.** Raw data lands in bronze and can never be queried. Cleaning writes silver, pending approval.
  Only a steward's approval grants the sandbox role `SELECT` on the table; rejection and erasure revoke it.
- **Sandbox.** Member and generated SQL is parsed and validated (single read-only SELECT, allow-listed tables
  and functions, masked columns refused), regenerated from the validated syntax tree, row policies applied as
  filtered subqueries, then run in a `READ ONLY` transaction under `SET LOCAL ROLE dataforge_sandbox` with a
  statement timeout and a row cap. Every run is recorded as a sandbox session and counts toward quota.
- **Personal information** is detected by value and by column name, masked in every preview and in silver,
  blocked in SQL, and never reaches gold. Classification is raised automatically when PII is present.
- **Semantic layer.** A metric is defined once over column names (additive, count, distinct, or ratio) and
  applies to every approved dataset with those columns. The assistant, analysis, dashboards, monitors,
  predictions and Superset all compute it the same way.
- **Assistant.** Claude (when `ANTHROPIC_API_KEY` is set) or the rule-based planner proposes a *plan* (dataset,
  measure, method, grouping, filters, period) constrained by a JSON schema whose enums hold only the member's
  permitted fields. The plan is validated again, the analysis engine composes the SQL, and the sandbox runs it
  under the member's role. It answers single figures, comparisons and trends, explains a change in a total
  ("why did revenue fall last month?") as each group's contribution, and tests whether two measures move
  together. A follow-up ("what about Gauteng?", "and by course?") builds on the member's own previous answer
  when the client sends its id as `contextId`. Personal questions, forecasts, prompt injection and ungrounded
  questions are refused and recorded. The model never writes SQL and never sees data rows. Requests use
  `claude-opus-5-5` at low effort with server-side refusal fallbacks enabled (`fallbacks: "default"`); remove
  that parameter in `infrastructure/llm.py` if you do not want declined requests retried on another model.
- **Audit.** `app.audit_events` is append-only: a trigger rejects `UPDATE`, `DELETE` and `TRUNCATE` for everyone.
- **Erasure.** Erasing a dataset drops every layer's tables and landed files and returns a hashed receipt.
- **Landed files** are kept on local disk, or in Cloudinary when `CLOUDINARY_URL` is set. Cloudinary holds them
  as raw, authenticated assets (no public URL reaches them), encrypted with AES-256-GCM before they leave the
  API (`STORAGE_ENCRYPT`, keyed from `SECRETS_KEY`). Erasure deletes them there too.
- **Refresh.** A steward can have a connector re-read hourly, daily or weekly, or anyone allowed can press
  "Refresh now". Each dataset from it lands a new raw version (only when the source changed), reruns its latest
  applied recipe, and keeps its approval only if the cleaned columns, their types and the personal information
  are unchanged. Anything else goes back to a steward, who is notified. Each dataset refreshes in one
  transaction, so readers see the old data or the new.
- **Notifications.** Alerts, decisions on a member's data, data waiting for a steward, shares and refresh
  problems appear under the bell in the web app (`GET /api/v1/notifications`). Alerts are also emailed to the
  monitor's owner unless they turn it off.

## Running locally

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), PostgreSQL 14+ (tested on 18).

```bash
uv sync                               # creates .venv with runtime and dev dependencies
cp .env.example .env                  # then set DATABASE_URL and the secrets
uv run dataforge migrate              # schemas, tables, sandbox role, audit trigger
uv run dataforge seed-demo            # optional: demo organisation through the real pipeline (~20 s)
uv run dataforge serve --reload       # http://127.0.0.1:8000, docs at /docs when DEBUG=true
```

The database login needs `CREATEROLE` the first time migrations run, because they create the
`dataforge_sandbox` role and grant it to the application's own login.

Demo members (with `DEMO_MODE=true`) sign in as `thandi@dataforge.local` (analyst), `sipho@` (practitioner),
`naledi@` (steward), `lerato@` (executive), `amina@` (admin), `johan@` (auditor) and `zola@` (curator), using
`DEMO_PASSWORD`, or with one click through `POST /api/v1/auth/demo {"role": "analyst"}`. Turn demo mode off
in production (the settings refuse to start otherwise).

### The web app

The web app in [`../skills-quest-forge`](../skills-quest-forge) is the client for this API. Run it with
`npm install && npm run dev` (http://localhost:5173); it reads the API's address from `VITE_API_URL`.

- `CORS_ORIGINS` must include the web app's origin, and `FRONTEND_URL` is where emailed links lead
  (`/verify`, `/reset-password`). Both default to `http://localhost:5173`.
- Sessions are an access token (short-lived, in the response body, sent as `Authorization: Bearer`) and a
  rotating refresh token that only ever travels in an HttpOnly cookie scoped to `/api/v1/auth`, so scripts in
  the page cannot read it. The web app calls the API with credentials included. When the web app and the API
  are on different sites, set `REFRESH_COOKIE_SAMESITE=none`. Changing or resetting a password ends every
  session.
- Email addresses are checked for shape only, so internal domains such as `name@org.local` work. Whether an
  address is real is settled by the verification email.
- Listing endpoints answer with what the member may see, which can be nothing: an auditor gets an empty
  analysis catalogue rather than a refusal. Actions the role may not take are refused with `403`.
- Reading never writes: a member who cannot clean a dataset sees its latest recipe, and no draft is started.
- Superset is kept in step after approvals, cleaning runs, policy changes and metric publishing. Those syncs run
  after the response, so the action never waits on Superset; failures are logged as `superset_sync_failed`.
- `POST /api/v1/contact` (public, rate-limited) emails the public contact form to `CONTACT_EMAIL`.

### Commands

| Command | What it does |
| --- | --- |
| `dataforge migrate` | Apply migrations |
| `dataforge seed-demo` | Create demo members, datasets, metrics, dashboards, monitors, a model version |
| `dataforge create-admin` | Create or promote an administrator |
| `dataforge evaluate-monitors` | Evaluate due monitors once (for cron when `MONITOR_INTERVAL_SECONDS=0`) |
| `dataforge refresh-connectors` | Refresh connectors whose schedule is due, once (for cron when `MONITOR_INTERVAL_SECONDS=0`) |
| `dataforge superset-sync` | Create the read-only BI login and sync datasets, metrics and row policies to Superset |
| `dataforge serve` | Run the API |

## Superset and Jupyter

Each has its own virtual environment under `integrations/`, so their dependencies never mix with the API's.

**Superset** (Python 3.11):

```bash
cd integrations/superset
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -c constraints-6.1.0.txt   # .venv/bin/python on Linux/macOS
cp .env.example .env                       # SSO_SHARED_SECRET and SUPERSET_ADMIN_PASSWORD must match the API's .env
.venv/Scripts/python bootstrap.py          # migrations, service account, roles (idempotent)
.venv/Scripts/python run.py                # http://localhost:8088
cd ../.. && uv run dataforge superset-sync # register the warehouse, approved datasets, metrics and row policies
```

- Members open Superset from DataForge (`POST /api/v1/integrations/superset/sign-in`): a two-minute, single-use
  token signed with `SSO_SHARED_SECRET` signs them in with roles mapped from their DataForge role
  (auditors are refused). There is no second password.
- Superset reads the warehouse through `dataforge_bi`, a read-only login that inherits the sandbox role's grants:
  approved silver and gold tables only, `default_transaction_read_only`, a statement timeout. Only
  organisation-wide datasets are registered, each with the semantic layer's metrics; row policies become
  Superset row-level security rules for `df_<role>` roles.
- Embedded dashboards use guest tokens that carry the viewer's row policies
  (`POST /api/v1/integrations/superset/dashboards/{id}/guest-token`).
- The UI uses the DataForge design system (`dataforge_superset/theme.py`): ivory surfaces, charcoal controls,
  vermilion as a signal, Instrument Sans and Fraunces, pill controls, DataForge chart colours and logo.
  Dark mode and theme swapping are off, matching DataForge.

**Jupyter** (Python 3.12):

```bash
cd integrations/jupyter
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
cp .env.example .env                       # SSO_SHARED_SECRET and JUPYTER_API_TOKEN must match the API's .env
.venv/Scripts/python run.py                # http://localhost:8888/jupyter
```

- Members with code roles open JupyterLab from DataForge (`POST /api/v1/integrations/jupyter/sign-in`).
  They work in `members/<their id>/` and `shared/`; other members' folders and terminals are refused.
- Kernels hold no database credentials. They start with the member's own notebook token and query through
  the API with the `dataforge` library, so permissions, masking and row policies apply:
  `import dataforge as df; df.sql("SELECT ...")`, or `%load_ext dataforge` and a `%%sql` cell.
- Any DataForge notebook or generated SQL can be opened in Jupyter
  (`POST /api/v1/integrations/jupyter/notebooks/{id}`, `POST /api/v1/integrations/jupyter/sql`).
- The UI uses the DataForge theme (`dataforge_jupyter/custom/custom.css`, settings overrides, logo, title).

## Tests and checks

```bash
uv run pytest            # unit tests, plus integration tests on a fresh <database>_test database
uv run ruff check . && uv run ruff format --check .
uv run mypy src          # strict
```

Integration tests create (on first run), drop and rebuild the `_test` database next to `DATABASE_URL` (override with
`DATAFORGE_TEST_DATABASE_URL`), run the migrations, seed through the real pipeline and exercise the API:
registration, MFA, refresh-token reuse detection, lockout, the full upload → clean → approve → analyse path,
reproducible reruns, masking, row policies, dashboards and embedding, alerts and root cause, predictions and
exports, notebooks, search, erasure and the append-only audit trail. `test_refresh.py` refreshes a real
connector (a schema in the test database) and checks that approval is kept or sent back for review, and that
the right members are notified; `test_assistant.py` covers explaining changes, relationships and follow-ups.
The Claude request itself is checked against a fake client (`tests/unit/test_llm.py`), and the worker pool and
Cloudinary store have their own unit tests. GitHub Actions runs all of this, plus the Docker image builds,
on every push (`.github/workflows/ci.yml`). `test_frontend_contract.py` covers what
the web app relies on: viewers never start a draft recipe, a skipped step always saves, clearing a quota returns
to the role's default, sharing can be withdrawn, exports cannot carry spreadsheet formulas, and the contact form.

## Docker

`docker-compose.yml` runs PostgreSQL, migrations, the API, Superset and Jupyter together. Copy `.env.example`
to `.env`, set the secrets named in the compose file, then `docker compose up --build`.

## Limits to know about

- **Sandbox isolation.** SQL runs under a least-privilege database role with caps. Parsing, profiling and full
  cleaning runs happen in worker processes (`SANDBOX_WORKERS` at once) that are killed at their time limit, with
  a memory cap on Linux (`SANDBOX_WORKER_MEMORY_MB`); previews on a 400-row sample run in threads. A request
  still waits for its job to finish, and jobs do not survive a restart: a durable queue with a status endpoint
  is the next step for very large files. Container isolation per session (gVisor) is an infrastructure concern
  and is not part of this service.
- **Raw history.** Every changed refresh keeps a new bronze version for lineage. There is no retention policy
  yet, so a source that changes every hour grows bronze every hour.
- **Jupyter isolation.** One Jupyter server serves all members. Folders, kernels and data access are separated,
  but kernels share the server's operating-system user. For hard isolation between members, run JupyterHub
  with a container spawner and the same `dataforge` library.
- **Demo connectors** (`Service desk`, `Analytics warehouse`) point at hosts that do not exist; their datasets are
  seeded directly. Real connectors are tested live when created or approved.
- **Docker images** are defined and the compose file validates, but the images were not built in this
  environment (the Docker daemon was not running).
