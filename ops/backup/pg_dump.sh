#!/usr/bin/env bash
# Dump the DataForge Postgres database. Pass DATABASE_URL or PG* env vars.
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-./var/backups}"
mkdir -p "$OUT_DIR"
OUT_FILE="${OUT_DIR}/dataforge-${STAMP}.dump"

if [[ -z "${DATABASE_URL:-}" && -z "${PGDATABASE:-}" ]]; then
  echo "Set DATABASE_URL (postgresql://...) or standard PG* variables." >&2
  exit 1
fi

# Accept SQLAlchemy-style URLs by stripping the +asyncpg driver marker.
URL="${DATABASE_URL:-}"
URL="${URL/postgresql+asyncpg:\/\//postgresql:\/\/}"

if [[ -n "$URL" ]]; then
  pg_dump --format=custom --file="$OUT_FILE" "$URL"
else
  pg_dump --format=custom --file="$OUT_FILE"
fi

echo "Wrote $OUT_FILE"
ls -lh "$OUT_FILE"
