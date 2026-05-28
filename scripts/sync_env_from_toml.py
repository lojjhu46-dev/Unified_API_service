"""将 .env.toml 同步为当前应用可读取的 .env"""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TOML_PATH = ROOT / ".env.toml"
ENV_PATH = ROOT / ".env"


def _cfg(data: dict, section: str, key: str, default: str = "") -> str:
    value = data.get(section, {}).get(key, default)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> int:
    if not TOML_PATH.exists():
        print(f"未找到 {TOML_PATH}")
        return 1

    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    lines = [
        "# Auto-generated from .env.toml. Do not commit real secrets.",
        "",
        f"APP_NAME={_cfg(data, 'app', 'app_name', 'Unified_API_Service')}",
        f"APP_VERSION={_cfg(data, 'app', 'app_version', '0.1.0')}",
        "APP_DEBUG=true",
        "APP_HOST=0.0.0.0",
        "APP_PORT=8000",
        "",
        f"LLM_PROVIDER={_cfg(data, 'llm', 'provider', 'deepseek')}",
        f"DEEPSEEK_API_KEY={_cfg(data, 'llm', 'deepseek_api_key')}",
        f"DEEPSEEK_BASE_URL={_cfg(data, 'llm', 'deepseek_base_url', 'https://api.deepseek.com')}",
        f"DEEPSEEK_MODEL={_cfg(data, 'llm', 'deepseek_model', 'deepseek-chat')}",
        "LLM_TIMEOUT_SECONDS=30",
        "",
        f"SERPER_API_KEY={_cfg(data, 'search', 'serper_api_key')}",
        f"SERPER_URL={_cfg(data, 'search', 'serper_url', 'https://google.serper.dev/search')}",
        "SEARCH_TIMEOUT=10",
        "MAX_RETRIES=2",
        "",
        "CHROMA_PERSIST_DIR=./data/chroma",
        "UPLOAD_DIR=./data/uploads",
        "CHUNK_SIZE=500",
        "CHUNK_OVERLAP=50",
        "EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2",
        "",
        "MEMORY_WINDOW_MESSAGES=6",
        "MEMORY_MAX_MESSAGES=20",
        "",
        f"FEISHU_APP_ID={_cfg(data, 'feishu', 'app_id')}",
        f"FEISHU_APP_SECRET={_cfg(data, 'feishu', 'app_secret')}",
        f"FEISHU_VERIFICATION_TOKEN={_cfg(data, 'feishu', 'verification_token')}",
        f"FEISHU_ENCRYPT_KEY={_cfg(data, 'feishu', 'encrypt_key')}",
        "",
        "LOG_LEVEL=INFO",
        "LOG_FORMAT=json",
        "LOG_FILE=./logs/app.log",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {ENV_PATH}。请重启服务后再运行 smoke 验收。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
