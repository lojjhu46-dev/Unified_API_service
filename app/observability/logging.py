"""日志模块"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from app.config import settings


THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "chromadb",
    "uvicorn.access",
)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        for key in (
            "request_id",
            "user_id",
            "channel",
            "route",
            "latency_ms",
            "component",
            "storage_backend",
            "storage_degraded",
            "fallback_reason",
            "error",
            "encrypted",
            "event_type",
            "type",
            "body_keys",
            "event_id",
            "message_id",
            "chat_type",
            "dedupe_key",
            "dedupe_key_source",
            "dedupe_result",
            "dedupe_backend",
            "dedupe_degraded",
            "dedupe_ttl_seconds",
            "feishu_retry_likely",
            "recent_answer_reuse_hit",
            "recent_answer_reuse_miss_reason",
            "recent_answer_reuse_similarity",
            "recent_answer_reuse_best_similarity",
            "recent_answer_reuse_turn_offset",
            "recent_answer_reuse_candidate_count",
            "recent_answer_reuse_threshold",
            "recent_answer_reuse_min_chars",
            "reply_ok",
        ):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging():
    """配置日志"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_build_formatter())
    root_logger.addHandler(console_handler)

    if settings.log_file and not _disable_file_logging():
        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_build_formatter())
        root_logger.addHandler(file_handler)

    _configure_third_party_loggers()


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


def _build_formatter() -> logging.Formatter:
    if settings.log_format == "json":
        return JSONFormatter()
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _disable_file_logging() -> bool:
    if not settings.log_disable_file_in_tests:
        return False
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or "pytest" in os.path.basename(sys.argv[0])
    )


def _configure_third_party_loggers() -> None:
    level = getattr(logging, settings.log_third_party_level.upper(), logging.WARNING)
    for logger_name in THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


setup_logging()
