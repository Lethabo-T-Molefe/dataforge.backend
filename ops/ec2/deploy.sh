#!/bin/bash
# Deploys one commit of dataforge.backend on the EC2 host, then checks health and rolls back if it fails.
# CI runs it through AWS Systems Manager after the checks pass on main:  /opt/dataforge/deploy.sh <commit sha>
# It runs as root; git and uv run as the dataforge user that owns /opt/dataforge.
# Migrations run when the service starts (ExecStartPre); a rollback restores the code, not the database schema.
set -euo pipefail

SHA="${1:?usage: deploy.sh <commit sha>}"
APP=/opt/dataforge/app

as_app() { sudo -u dataforge -H "$@"; }
sync_deps() { as_app env UV_PYTHON_INSTALL_DIR=/opt/dataforge/python /usr/local/bin/uv sync --frozen --no-dev; }
checkout() { as_app git -C "$APP" fetch --depth 1 origin "$1" && as_app git -C "$APP" reset --hard FETCH_HEAD; }
healthy() {
  for _ in $(seq 1 36); do
    curl -fsS -m 5 http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

PREV=$(as_app git -C "$APP" rev-parse HEAD)
echo "Deploying ${SHA} (running ${PREV})"

checkout "$SHA"
cd "$APP"
sync_deps
systemctl restart dataforge

if healthy; then
  echo "Deployed ${SHA}"
  exit 0
fi

echo "Health check failed for ${SHA}; recent service log:" >&2
journalctl -u dataforge --no-pager -n 40 -o cat | cut -c1-300 >&2 || true

echo "Rolling back to ${PREV}" >&2
checkout "$PREV"
sync_deps
systemctl restart dataforge
if healthy; then echo "Rolled back to ${PREV}" >&2; else echo "Rollback is unhealthy too; check the server" >&2; fi
exit 1
