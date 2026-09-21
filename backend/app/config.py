from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "CottonLens AI"
    database_url: str = "sqlite:///./cottonlens.db"
    artifact_dir: str = "../runtime/artifacts/current"
    demo_mode: bool = True
    cors_origins: list[str] = ["http://localhost:4200", "http://localhost:8080"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

