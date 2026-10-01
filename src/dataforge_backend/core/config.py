from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT_SECRET = "dev-only-change-me-dev-only-change-me"


class Settings(BaseSettings):
    """All runtime configuration, read from the environment or a .env file."""

    PROJECT_NAME: str = "DataForge.Backend"
    ENVIRONMENT: Literal["development", "test", "production"] = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/dataforge"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8080",
            "http://localhost:5173",
            "http://localhost:4173",
            "http://127.0.0.1:5173",
        ]
    )
    FRONTEND_URL: str = "http://localhost:8080"

    # Identity
    JWT_SECRET: str = _INSECURE_DEFAULT_SECRET
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str = "dataforge"
    ACCESS_TOKEN_MINUTES: int = 15
    REFRESH_TOKEN_DAYS: int = 14
    MFA_CHALLENGE_MINUTES: int = 5
    EMAIL_TOKEN_HOURS: int = 48
    PASSWORD_MIN_LENGTH: int = 10
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCK_MINUTES: int = 15
    # Encrypts connector credentials and MFA secrets at rest. Falls back to a key derived from JWT_SECRET.
    SECRETS_KEY: str | None = None

    # Email (verification). Without SMTP_HOST, emails are written to the log instead of sent.
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM: str = "DataForge <no-reply@dataforge.local>"
    SMTP_STARTTLS: bool = True

    # Ingestion
    STORAGE_DIR: Path = Path("var/landing")
    MAX_UPLOAD_MB: int = 250

    # Sandbox execution
    SANDBOX_ROLE: str = "dataforge_sandbox"
    SANDBOX_STATEMENT_TIMEOUT_MS: int = 15_000
    SANDBOX_MAX_ROWS: int = 5_000

    # AI assistant. Without an API key the assistant uses its rule-based parser.
    ANTHROPIC_API_KEY: str | None = None
    ASSISTANT_MODEL: str = "claude-opus-5"
    ASSISTANT_USE_LLM: bool = True
    ASSISTANT_TIMEOUT_SECONDS: float = 30.0

    # Monitoring. 0 disables the in-process scheduler (use `dataforge evaluate-monitors` from cron instead).
    MONITOR_INTERVAL_SECONDS: int = 900

    # Rate limiting. With REDIS_URL set, limits are shared across API replicas.
    REDIS_URL: str | None = None
    RATE_LIMIT_DEFAULT: str = "600/minute"

    # Integrations
    SLACK_SIGNING_SECRET: str | None = None
    SLACK_WEBHOOK_URL: str | None = None
    SENTRY_DSN: str | None = None
    PUBLIC_API_URL: str = "http://localhost:8000"

    # Single sign-on into Superset and Jupyter. A separate secret, so neither service ever holds JWT_SECRET.
    SSO_SHARED_SECRET: str = "dev-only-sso-secret-change-me-dev-only"
    SSO_TOKEN_SECONDS: int = 120
    NOTEBOOK_TOKEN_HOURS: int = 8

    # Apache Superset (dashboards, Explore, SQL Lab). Empty URL disables the integration.
    SUPERSET_URL: str | None = "http://localhost:8088"
    SUPERSET_PUBLIC_URL: str | None = None  # the URL browsers use, if different from SUPERSET_URL
    SUPERSET_ADMIN_USER: str = "dataforge-service"
    SUPERSET_ADMIN_PASSWORD: str = "change-me-superset-service"
    SUPERSET_DATABASE_NAME: str = "DataForge warehouse"
    # The read-only login Superset uses. It inherits the sandbox role's grants: approved tables only.
    BI_DB_ROLE: str = "dataforge_bi"
    BI_DB_PASSWORD: str = "change-me-bi-reader"

    # JupyterLab (notebooks). Empty URL disables the integration.
    JUPYTER_URL: str | None = "http://localhost:8888/jupyter"
    JUPYTER_PUBLIC_URL: str | None = None
    JUPYTER_API_TOKEN: str = "change-me-jupyter-service-token"

    # Demo: one-click demo sign-in and role preview for the prototype frontend. Never enable in production.
    DEMO_MODE: bool = True
    DEMO_PASSWORD: str = "DataForge!2026"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_production_safety(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if self.JWT_SECRET == _INSECURE_DEFAULT_SECRET or len(self.JWT_SECRET) < 32:
                raise ValueError(
                    "JWT_SECRET must be set to a random value of at least 32 characters in production"
                )
            if self.DEMO_MODE:
                raise ValueError("DEMO_MODE must be false in production")
            if self.DEBUG:
                raise ValueError("DEBUG must be false in production")
            for name in (
                "SSO_SHARED_SECRET",
                "SUPERSET_ADMIN_PASSWORD",
                "BI_DB_PASSWORD",
                "JUPYTER_API_TOKEN",
            ):
                value = getattr(self, name)
                if "change-me" in value or len(value) < 24:
                    raise ValueError(
                        f"{name} must be set to a random value of at least 24 characters in production"
                    )
        return self

    @cached_property
    def sync_database_url(self) -> str:
        """The same database through the plain asyncpg DSN form (used for raw driver connections)."""
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

    @property
    def llm_enabled(self) -> bool:
        return self.ASSISTANT_USE_LLM and bool(self.ANTHROPIC_API_KEY)


settings = Settings()
