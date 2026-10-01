"""Superset configuration for DataForge.

Superset is DataForge's dashboard, Explore and SQL Lab surface. It reads the warehouse through a read-only
login that inherits the sandbox role's grants (approved silver and gold tables only), members sign in
from DataForge, and the UI uses the DataForge design system.

Settings come from the environment (see .env.example in this folder).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataforge_superset.sso import blueprint  # noqa: E402
from dataforge_superset.theme import CATEGORICAL_SCHEME, SEQUENTIAL_SCHEME, dataforge_theme  # noqa: E402


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Set {name} in the environment (see integrations/superset/.env.example).")
    return value


# Metadata (dashboards, charts, users) lives in its own database, next to DataForge's.
SQLALCHEMY_DATABASE_URI = _env("SUPERSET_METADATA_URL", "postgresql+psycopg2://postgres@localhost:55432/superset")
SECRET_KEY = _env("SUPERSET_SECRET_KEY")

APP_NAME = "DataForge"
APP_ICON = "/dataforge-static/logo-primary.png"
FAVICONS = [
    {"href": "/dataforge-static/favicon-32.png", "sizes": "32x32", "type": "image/png"},
    {"href": "/dataforge-static/apple-touch-icon.png", "rel": "apple-touch-icon", "sizes": "180x180"},
]
LOGO_TARGET_PATH = "/superset/welcome/"
LOGO_TOOLTIP = "DataForge analytics"

# The DataForge design system. DataForge has one (light) theme, so dark mode is off and admins
# cannot swap in another system theme.
THEME_DEFAULT = dataforge_theme()
THEME_DARK = None
ENABLE_UI_THEME_ADMINISTRATION = False
EXTRA_CATEGORICAL_COLOR_SCHEMES = [CATEGORICAL_SCHEME]
EXTRA_SEQUENTIAL_COLOR_SCHEMES = [SEQUENTIAL_SCHEME]

# DataForge single sign-on and brand assets.
BLUEPRINTS = [blueprint]
AUTH_USER_REGISTRATION = False
PUBLIC_ROLE_LIKE = None

FEATURE_FLAGS = {
    "EMBEDDED_SUPERSET": True,  # dashboards embedded in DataForge with guest tokens
    "DASHBOARD_RBAC": True,
    "ENABLE_TEMPLATE_PROCESSING": False,  # no Jinja in SQL: queries mean what they say
    "ALERT_REPORTS": False,  # alerts are DataForge monitors
}

# Embedded dashboards: DataForge's backend requests guest tokens with row-level security clauses.
GUEST_ROLE_NAME = "DataForge Guest"
GUEST_TOKEN_JWT_SECRET = _env("SUPERSET_GUEST_TOKEN_SECRET")
GUEST_TOKEN_JWT_EXP_SECONDS = 300

# SQL Lab: read-only, capped, and never allowed to write or upload.
SQLLAB_TIMEOUT = 30
SQLLAB_ASYNC_TIME_LIMIT_SEC = 60
SQL_MAX_ROW = 50_000
DISPLAY_MAX_ROW = 5_000
PREVENT_UNSAFE_DB_CONNECTIONS = True
CSV_EXPORT = {"encoding": "utf-8"}
ALLOWED_USER_CSV_SCHEMA_FUNC = lambda database, user: []  # noqa: E731

# Frontend origins that may embed Superset.
_FRONTEND = [o.strip() for o in _env("DATAFORGE_FRONTEND_ORIGINS", "http://localhost:8080,http://localhost:5173").split(",")]
TALISMAN_ENABLED = True
TALISMAN_CONFIG = {
    "content_security_policy": {
        "base-uri": ["'self'"],
        "default-src": ["'self'"],
        "img-src": ["'self'", "blob:", "data:"],
        "worker-src": ["'self'", "blob:"],
        "connect-src": ["'self'"],
        "object-src": "'none'",
        "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
        "font-src": ["'self'", "https://fonts.gstatic.com"],
        "script-src": ["'self'", "'strict-dynamic'"],
        "frame-ancestors": ["'self'", *_FRONTEND],
    },
    "content_security_policy_nonce_in": ["script-src"],
    "frame_options": None,
    "force_https": os.environ.get("SUPERSET_FORCE_HTTPS", "false") == "true",
    "session_cookie_secure": os.environ.get("SUPERSET_FORCE_HTTPS", "false") == "true",
}
TALISMAN_DEV_CONFIG = TALISMAN_CONFIG
SESSION_COOKIE_SAMESITE = "None" if os.environ.get("SUPERSET_FORCE_HTTPS") == "true" else "Lax"

WTF_CSRF_ENABLED = True
WTF_CSRF_EXEMPT_LIST = ["superset.views.core.log", "superset.charts.data.api.data"]

CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}
DATA_CACHE_CONFIG = CACHE_CONFIG
RATELIMIT_STORAGE_URI = os.environ.get("REDIS_URL", "memory://")
