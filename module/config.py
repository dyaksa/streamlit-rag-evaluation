from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Config(BaseSettings):
    APP_PORT: int = Field(
        default=8501, alias="APP_PORT", description="Port number for the application"
    )
    APP_HOST: str = Field(
        default="0.0.0.0",
        alias="APP_HOST",
        description="Host address for the application",
    )
    OPEN_ROUTER_MODEL: str = Field(
        default="gpt-4o-mini",
        alias="OPEN_ROUTER_MODEL",
        description="Name of the LLM model to use",
    )
    OPEN_ROUTER_API_KEY: str = Field(
        default="",
        alias="OPEN_ROUTER_API_KEY",
        description="API key for the LLM model",
    )
    OPEN_ROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPEN_ROUTER_BASE_URL",
        description="Base URL for the LLM model API",
    )
    HUGGINGFACE_EMBEDDING_MODEL: str = Field(
        default="",
        alias="HUGGINGFACE_EMBEDDING_MODEL",
        description="Huggingface model for embedding",
    )
    HUGGINGFACE_API_KEY: str = Field(
        default="", alias="HUGGINGFACE_API_KEY", description="Huggingface api key"
    )

    HUGGINGFACE_LLM_MODEL: str = Field(
        default="",
        alias="HUGGINGFACE_LLM_MODEL",
        description="Huggingface model for LLM",
    )

    GEMINI_API_KEY: str = Field(
        default="", alias="GEMINI_API_KEY", description="Gemini API key"
    )
    GEMINI_MODEL_LLM: str = Field(
        default="", alias="GEMINI_MODEL_LLM", description="Gemini model"
    )
    GEMINI_MODEL_EMBEDDING: str = Field(
        default="",
        alias="GEMINI_MODEL_EMBEDDING",
        description="Gemini embedding model",
    )

    MISTRAL_API_KEY: str = Field(
        default="", alias="MISTRAL_API_KEY", description="Mistral API key"
    )
    MISTRAL_LLM_MODEL: str = Field(
        default="", alias="MISTRAL_LLM_MODEL", description="Mistral LLM model"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


@lru_cache()
def get_settings() -> Config:
    return Config()


settings = get_settings()
