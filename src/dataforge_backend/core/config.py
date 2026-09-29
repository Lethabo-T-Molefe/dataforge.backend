from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DataForge.Backend"
    DEBUG: bool = True
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/dataforge"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
