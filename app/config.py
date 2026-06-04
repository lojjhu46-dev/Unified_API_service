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
    redis_connect_timeout: int = 5
    redis_health_check_interval: int = 30
    redis_retry_cooldown_seconds: int = 10

    # Serper 搜索配置
    serper_api_key: Optional[str] = None
    serper_url: str = "https://google.serper.dev/search"
    search_timeout: int = 10
    max_retries: int = 2

    # 文档处理配置
    upload_dir: str = "./data/uploads"
    personal_upload_dir: str = "./data/personal_uploads"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 会话配置
    memory_window_messages: int = 6
    memory_max_messages: int = 20
    memory_session_ttl_seconds: int = 7 * 24 * 60 * 60
    recent_answer_reuse_similarity_threshold: float = 0.8
    recent_answer_reuse_min_chars: int = 6

    # 检索配置
    top_k: int = 5
    bm25_top_k: int = 3
    hybrid_vector_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    global_max_chunks: int = 50
    rag_neighbor_window: int = 1
    rag_global_neighbor_window: int = 4
    rag_context_max_chars: int = 8000
    rag_display_snippet_chars: int = 200
    personal_kb_strict_owner_filter: bool = True
    rag_parallel_subquestion_max: int = 4
    rag_llm_subquestion_split_enabled: bool = True

    # OpenSearch 检索配置
    opensearch_enabled: bool = False
    opensearch_url: str = "http://localhost:9201"
    opensearch_index_name: str = "unified_kb_chunks"
    opensearch_username: Optional[str] = None
    opensearch_password: Optional[str] = None
    opensearch_timeout_seconds: int = 3
    opensearch_lexical_top_k: int = 80
    opensearch_rrf_k: int = 60
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_use_ik_analyzer: bool = True

    # 飞书配置
    feishu_app_id: Optional[str] = None
    feishu_app_secret: Optional[str] = None
    feishu_verification_token: Optional[str] = None
    feishu_encrypt_key: Optional[str] = None
    feishu_heartbeat_enabled: bool = True
    feishu_heartbeat_initial_delay_seconds: int = 5
    feishu_heartbeat_interval_seconds: int = 20
    feishu_heartbeat_max_count: int = 12
    feishu_pending_file_ttl_seconds: int = 30 * 60

    # 安全配置
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    cors_allowed_origins: str = ""
    cors_allow_credentials: bool = False
    cors_allowed_methods: str = "GET,POST,DELETE,OPTIONS"
    cors_allowed_headers: str = "Authorization,Content-Type,X-API-Key,X-User-Id,X-Channel"

    # 限流配置
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "./logs/app.log"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3
    log_third_party_level: str = "WARNING"
    log_disable_file_in_tests: bool = True


def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def parse_csv_setting(value: str) -> list[str]:
    """Parse comma-separated env settings into a clean list."""
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def ensure_directories():
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.personal_upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    if settings.log_file:
        os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)


ensure_directories()
