"""阶段七前置真实 API 可用性验收脚本"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".env.toml"
REPORT_DIR = ROOT / "reports"


@dataclass
class CheckResult:
    name: str
    status: str
    elapsed_ms: float
    detail: str


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _cfg(config: dict, section: str, key: str, default=None):
    return config.get(section, {}).get(key, default)


def _has_value(value: str | None) -> bool:
    return bool(value and value.strip())


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    start = time.perf_counter()
    try:
        detail = fn()
        status = "PASS"
    except SkipCheck as exc:
        detail = str(exc)
        status = "SKIPPED"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        status = "FAIL"
    elapsed_ms = (time.perf_counter() - start) * 1000
    return CheckResult(name=name, status=status, elapsed_ms=elapsed_ms, detail=detail)


class SkipCheck(Exception):
    """跳过当前验收项"""


class SmokeRunner:
    def __init__(self, config: dict, timeout: float):
        self.config = config
        self.base_url = os.getenv("SMOKE_BASE_URL") or _cfg(
            config, "app", "base_url", "http://127.0.0.1:8000"
        )
        self.public_base_url = _cfg(config, "app", "public_base_url", "")
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.session_id: str | None = None
        self.uploaded_phrase = "蓝鲸烟火苔原"

    def close(self) -> None:
        self.client.close()

    def check_health(self) -> str:
        resp = self.client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise AssertionError(f"health status unexpected: {data}")
        return f"status=ok version={data.get('version')}"

    def check_root(self) -> str:
        resp = self.client.get("/")
        resp.raise_for_status()
        data = resp.json()
        if data.get("docs") != "/docs":
            raise AssertionError(f"root docs unexpected: {data}")
        return f"service={data.get('service')} docs=/docs"

    def check_direct(self) -> str:
        data = self._ask({"user_id": "smoke_user", "question": "你好"})
        if data.get("route") != "direct":
            raise AssertionError(f"route expected direct, got {data.get('route')}")
        self.session_id = data.get("session_id")
        return f"route=direct session_id={self.session_id}"

    def check_tool(self) -> str:
        data = self._ask({"user_id": "smoke_user", "question": "计算 2+3 等于多少"})
        if data.get("route") != "tool":
            raise AssertionError(f"route expected tool, got {data.get('route')}")
        if "5" not in data.get("answer", ""):
            raise AssertionError(f"answer missing result 5: {data.get('answer')}")
        return f"route=tool answer={data.get('answer')}"

    def check_deepseek(self) -> str:
        if not _cfg(self.config, "smoke", "enable_deepseek", True):
            raise SkipCheck("smoke.enable_deepseek=false")
        if not _has_value(_cfg(self.config, "llm", "deepseek_api_key", "")):
            raise SkipCheck("llm.deepseek_api_key 未配置")
        data = self._ask({"user_id": "smoke_user", "question": "用一句话介绍你自己"})
        answer = data.get("answer", "")
        if not answer:
            raise AssertionError("DeepSeek answer is empty")
        if "[MOCK回答]" in answer:
            raise AssertionError("DeepSeek check returned mock answer，请确认服务启动前已将 .env.toml 同步到 .env")
        return f"route={data.get('route')} answer_len={len(answer)}"

    def check_upload_and_rag(self) -> str:
        content = (
            f"阶段七前置验收文档。唯一短语：{self.uploaded_phrase}。"
            "如果被问到这个短语，请说明它来自 smoke 上传文档。"
        ).encode("utf-8")
        files = {"file": ("smoke_stage7.txt", content, "text/plain")}
        resp = self.client.post("/documents/upload", files=files)
        resp.raise_for_status()
        upload_data = resp.json()
        if upload_data.get("status") != "success":
            raise AssertionError(f"upload failed: {upload_data}")

        data = self._ask({
            "user_id": "smoke_user",
            "question": f"{self.uploaded_phrase} 来自哪里？",
            "need_web": "never",
            "top_k": 5,
        })
        if data.get("route") != "rag":
            raise AssertionError(f"route expected rag, got {data.get('route')}")
        if not data.get("sources"):
            raise AssertionError("RAG sources is empty")
        return f"document_id={upload_data.get('document_id')} sources={len(data.get('sources', []))}"

    def check_session_memory(self) -> str:
        first = self._ask({
            "user_id": "smoke_user",
            "session_id": self.session_id,
            "question": "请记住：我的测试代号是蓝鲸。",
            "need_web": "never",
        })
        session_id = first.get("session_id")
        second = self._ask({
            "user_id": "smoke_user",
            "session_id": session_id,
            "question": "我的测试代号是什么？",
            "need_web": "never",
        })
        if second.get("session_id") != session_id:
            raise AssertionError("session_id changed between turns")
        history = self.client.get(f"/sessions/{session_id}/history")
        history.raise_for_status()
        messages = history.json().get("messages", [])
        if len(messages) < 4:
            raise AssertionError(f"history too short: {messages}")
        return f"session_id={session_id} messages={len(messages)}"

    def check_serper_web(self) -> str:
        if not _cfg(self.config, "smoke", "enable_serper", True):
            raise SkipCheck("smoke.enable_serper=false")
        if not _has_value(_cfg(self.config, "search", "serper_api_key", "")):
            raise SkipCheck("search.serper_api_key 未配置")
        data = self._ask({
            "user_id": "smoke_user",
            "question": "今天人工智能新闻",
            "need_web": "always",
            "top_k": 3,
        })
        if data.get("route") != "web":
            raise AssertionError(f"route expected web, got {data.get('route')}")
        if not any(s.get("source_type") == "web_search" for s in data.get("sources", [])):
            raise AssertionError("web_search source missing，请确认服务启动前已将 .env.toml 同步到 .env")
        if not data.get("tool_trace"):
            raise AssertionError("tool_trace missing")
        return f"sources={len(data.get('sources', []))} tool_trace={len(data.get('tool_trace', []))}"

    def check_agentic_rag(self) -> str:
        if not _has_value(_cfg(self.config, "search", "serper_api_key", "")):
            raise SkipCheck("search.serper_api_key 未配置，跳过 agentic_rag 真实联网验收")
        data = self._ask({
            "user_id": "smoke_user",
            "question": "今天这个 smoke 文档短语有什么最新相关信息？",
            "top_k": 3,
        })
        if data.get("route") != "agentic_rag":
            raise AssertionError(f"route expected agentic_rag, got {data.get('route')}")
        if not data.get("sources"):
            raise AssertionError("agentic_rag sources is empty")
        return f"sources={len(data.get('sources', []))} route=agentic_rag"

    def check_feishu_config(self) -> str:
        if not _cfg(self.config, "smoke", "enable_feishu", True):
            raise SkipCheck("smoke.enable_feishu=false")
        has_feishu = all(
            _has_value(_cfg(self.config, "feishu", key, ""))
            for key in ["app_id", "app_secret", "verification_token"]
        )
        if not has_feishu:
            raise SkipCheck("飞书 app_id/app_secret/verification_token 暂未配置")
        if not _has_value(self.public_base_url):
            raise SkipCheck("app.public_base_url 暂未配置，无法做真实飞书回调验收")
        return f"public_base_url={self.public_base_url}"

    def _ask(self, payload: dict) -> dict:
        resp = self.client.post("/ask", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "answer" not in data:
            raise AssertionError(f"invalid ask response: {data}")
        return data


def _write_report(results: list[CheckResult], base_url: str) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"stage7_precheck_{ts}.md"
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    passed = [r for r in results if r.status == "PASS"]

    lines = [
        "# 阶段七前置验收报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- Base URL：{base_url}",
        f"- 结果汇总：PASS {len(passed)} / FAIL {len(failed)} / SKIPPED {len(skipped)}",
        "",
        "| 检查项 | 状态 | 耗时(ms) | 详情 |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        detail = result.detail.replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {result.name} | {result.status} | {result.elapsed_ms:.1f} | {detail} |"
        )
    lines.extend([
        "",
        "## 飞书人工验收",
        "",
        "- 当前脚本仅检查飞书配置是否齐备。",
        "- 如已配置 `public_base_url`，请在飞书开放平台将事件订阅 URL 指向 "
        "`{PUBLIC_BASE_URL}/channels/feishu/events` 后完成 challenge 与真实消息回复验收。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段七前置真实 API 可用性验收")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次 HTTP 请求超时时间")
    args = parser.parse_args()

    config = _load_config()
    runner = SmokeRunner(config=config, timeout=args.timeout)
    checks = [
        ("health", runner.check_health),
        ("root", runner.check_root),
        ("direct", runner.check_direct),
        ("tool", runner.check_tool),
        ("deepseek", runner.check_deepseek),
        ("upload_and_rag", runner.check_upload_and_rag),
        ("session_memory", runner.check_session_memory),
        ("serper_web", runner.check_serper_web),
        ("agentic_rag", runner.check_agentic_rag),
        ("feishu_config", runner.check_feishu_config),
    ]

    results: list[CheckResult] = []
    try:
        for name, fn in checks:
            result = _run_check(name, fn)
            results.append(result)
            print(f"[{result.status}] {name} ({result.elapsed_ms:.1f} ms) {result.detail}")
    finally:
        runner.close()

    report_path = _write_report(results, runner.base_url)
    print(f"\n报告已生成：{report_path}")

    required = {"health", "root", "direct", "tool", "upload_and_rag", "session_memory"}
    failed_required = [r for r in results if r.name in required and r.status != "PASS"]
    failed_real = [r for r in results if r.status == "FAIL"]
    if failed_required or failed_real:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
