"""Test configuration. Runs before the application is imported, so settings pick up the test environment."""

import os
from pathlib import Path

from dotenv import dotenv_values

_ENV = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
_DEV_URL = (
    os.environ.get("DATABASE_URL")
    or _ENV.get("DATABASE_URL")
    or "postgresql+asyncpg://postgres@localhost:5432/dataforge"
)

# Integration tests use their own database (dropped and rebuilt per run): DATAFORGE_TEST_DATABASE_URL from the
# environment or .env, so a hosted development database is never touched, or else one next to the development one.
os.environ["DATABASE_URL"] = (
    os.environ.get("DATAFORGE_TEST_DATABASE_URL")
    or _ENV.get("DATAFORGE_TEST_DATABASE_URL")
    or _DEV_URL.rsplit("/", 1)[0] + "/dataforge_test"
)
os.environ["ENVIRONMENT"] = "test"
os.environ["DEMO_MODE"] = "true"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["SUPERSET_URL"] = ""
os.environ["JUPYTER_URL"] = ""
os.environ["SLACK_WEBHOOK_URL"] = ""
os.environ["SMTP_HOST"] = ""
os.environ["MONITOR_INTERVAL_SECONDS"] = "0"
os.environ["STORAGE_DIR"] = str(Path(__file__).resolve().parents[1] / "var" / "test-landing")
# Tests never reach hosted services configured for development: files land on local disk, errors stay local.
os.environ["CLOUDINARY_URL"] = ""
os.environ["SENTRY_DSN"] = ""
