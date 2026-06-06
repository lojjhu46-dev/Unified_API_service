"""SQLite document registry for uploaded knowledge-base files."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings


class DocumentRegistry:
    """Small SQLite registry for document ownership and processing status."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.document_registry_db_path
        self._lock = Lock()

    def init(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    knowledge_base_type TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL DEFAULT '',
                    owner_open_id TEXT NOT NULL DEFAULT '',
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT '',
                    chat_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_owner "
                "ON documents (tenant_id, knowledge_base_type, owner_user_id)"
            )

    def create_processing(
        self,
        *,
        document_id: str,
        tenant_id: str,
        knowledge_base_type: str,
        owner_user_id: str = "",
        owner_open_id: str = "",
        original_filename: str,
        stored_filename: str,
        stored_path: str,
        channel: str = "",
        chat_id: str = "",
    ) -> None:
        self.init()
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, tenant_id, knowledge_base_type, owner_user_id,
                    owner_open_id, original_filename, stored_filename, stored_path,
                    channel, chat_id, status, chunk_count, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', 0, '', ?, ?)
                """,
                (
                    document_id,
                    tenant_id,
                    knowledge_base_type,
                    owner_user_id,
                    owner_open_id,
                    original_filename,
                    stored_filename,
                    stored_path,
                    channel,
                    chat_id,
                    now,
                    now,
                ),
            )

    def mark_ready(self, document_id: str, chunk_count: int) -> None:
        self._update_status(document_id, "ready", chunk_count=chunk_count, error="")

    def mark_failed(self, document_id: str, error: str | None = None) -> None:
        self._update_status(document_id, "failed", error=error or "")

    def get(self, document_id: str) -> dict[str, Any] | None:
        self.init()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def _update_status(
        self,
        document_id: str,
        status: str,
        *,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        self.init()
        updates = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, _utc_now()]
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            values.append(int(chunk_count))
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        values.append(document_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE document_id = ?",
                values,
            )

    def list_personal_ready(
        self,
        owner_user_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """列出指定用户的个人知识库 ready 文件，按创建时间倒序。"""
        self.init()
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, original_filename, stored_filename,
                       stored_path, created_at, updated_at
                FROM documents
                WHERE owner_user_id = ?
                  AND knowledge_base_type = 'personal'
                  AND status = 'ready'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (owner_user_id, safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_personal_ready_by_path(
        self,
        owner_user_id: str,
        stored_path: str,
    ) -> dict[str, Any] | None:
        """按 stored_path 查找当前用户 personal ready 文件，用于编辑前校验。"""
        self.init()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE stored_path = ?
                  AND owner_user_id = ?
                  AND knowledge_base_type = 'personal'
                  AND status = 'ready'
                LIMIT 1
                """,
                (stored_path, owner_user_id),
            ).fetchone()
        return dict(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


document_registry = DocumentRegistry()
