#!/usr/bin/env bash
# Restore a custom-format dump into the target database.
# WARNING: destructive. Prefer a empty or scratch database.
set -euo pipefail

DUMP_FILE="${1:-}"
if [[ -z "$DUMP_FILE" || ! -f "$DUMP_FILE" ]]; then
  echo "Usage: $0 path/to/dataforge-YYYYMMDD.dump" >&2
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" && -z "${PGDATABASE:-}" ]]; then
  echo "Set DATABASE_URL (postgresql://...) or standard PG* variables." >&2
  exit 1
fi

URL="${DATABASE_URL:-}"
URL="${URL/postgresql+asyncpg:\/\//postgresql:\/\/}"

echo "Restoring $DUMP_FILE (clean restore of app schema objects)…"
if [[ -n "$URL" ]]; then
  pg_restore --clean --if-exists --no-owner --no-acl --dbname="$URL" "$DUMP_FILE"
else
  pg_restore --clean --if-exists --no-owner --no-acl "$DUMP_FILE"
fi

echo "Restore finished. Run migrations if needed: cd dataforge.backend && uv run alembic upgrade head"
