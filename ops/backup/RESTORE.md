# Backup restore test

Do this at least once against a **non-production** database. Replication alone is not proof of recovery.

## What to back up

1. PostgreSQL (application schema, including rate buckets and sessions).
2. Uploaded files under `STORAGE_DIR` or Cloudinary assets referenced by the DB.
3. Secrets required to decrypt (`JWT_SECRET`, `SECRETS_KEY`, storage keys) — stored outside the dump.

## Procedure

1. Take a dump:

   ```bash
   export DATABASE_URL='postgresql://…'
   bash ops/backup/pg_dump.sh
   ```

2. Provision a scratch database (or drop/recreate a staging DB).

3. Restore:

   ```bash
   export DATABASE_URL='postgresql://…scratch…'
   bash ops/backup/pg_restore.sh var/backups/dataforge-….dump
   ```

4. Point a local API at the scratch DB, run `uv run alembic upgrade head` if the dump predates latest migrations, then:

   ```bash
   curl -sS "$API/api/v1/ready"
   curl -sS -o /dev/null -w "%{http_code}\n" "$API/api/v1/ready"
   ```

   Expect HTTP 200 and `"status":"ok"`.

5. Sign in (or demo sign-in) and open one dataset.

## Record of last successful restore

| Date (UTC) | Operator | Dump file | Notes |
|---|---|---|---|
| _pending_ | | | |

Update this table when you complete a restore drill (RPO default: 24h; RTO default: 4h — see [../slo.md](../slo.md)).
