"""统一配置模块"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 应用配置
    app_name: str = "Unified_API_Service"
    app_version: str = "0.1.0"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # DeepSeek API 配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: int = 30
    llm_provider: str = "mock"

    # Chroma 向量数据库配置
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Redis 配置
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "unified_rag"
    redis_socket_timeout: int = 5

    # Serper 搜索配置
    serper_api_key: Optional[str] = None
    serper_url: str = "https://google.serper.dev/search"
    search_timeout: int = 10
    max_retries: int = 2

    # 文档处理配置
    upload_dir: str = "./data/uploads"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 会话配置
    memory_window_messages: int = 6
    memory_max_messages: int = 20

    # 检索配置
    top_k: int = 5
    bm25_top_k: int = 3
    hybrid_vector_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    global_max_chunks: int = 50

    # 飞书配置
    feishu_app_id: Optional[str] = None
    feishu_app_secret: Optional[str] = None
    feishu_verification_token: Optional[str] = None
    feishu_encrypt_key: Optional[str] = None

    # 安全配置
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30

    # 限流配置
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "./logs/app.log"


def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def ensure_directories():
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)


ensure_directories()
