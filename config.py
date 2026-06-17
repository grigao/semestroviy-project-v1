from pydantic_settings import BaseSettings, SettingsConfigDict


# Настройки приложения
class Settings(BaseSettings):
    # Переменные
    debug: bool = False

    # Telegram Userbot
    app_id: int
    api_hash: str

    # Настройки Embedding
    embedding_model: str = "bge-m3"
    embedding_version: str = "v1"
    embedding_dimension: int = 1024
    lmstudio_base_url: str = "http://localhost:1234"
    lmstudio_timeout: int = 30
    embedding_batch_size: int = 10
    embedding_save_debug_vectors: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "blocks"

    vector_size: int = embedding_dimension

    # Настройки Pydantic
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

settings = Settings()
