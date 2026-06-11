"""Document registry tests."""

import pytest
from app.retrieval.document_registry import DocumentRegistry


def test_document_registry_create_mark_ready_and_get(tmp_path):
    registry = DocumentRegistry(str(tmp_path / "registry.sqlite3"))

    registry.create_processing(
        document_id="doc1",
        tenant_id="tenant_a",
        knowledge_base_type="personal",
        owner_user_id="user_a",
        owner_open_id="user_a",
        original_filename="a.txt",
        stored_filename="doc1_a.txt",
        stored_path="/tmp/doc1_a.txt",
        channel="api",
        chat_id="",
    )
    registry.mark_ready("doc1", 3)

    row = registry.get("doc1")
    assert row is not None
    assert row["tenant_id"] == "tenant_a"
    assert row["knowledge_base_type"] == "personal"
    assert row["owner_user_id"] == "user_a"
    assert row["status"] == "ready"
    assert row["chunk_count"] == 3
    assert row["error"] == ""
    assert row["is_generated_copy"] == 0
    assert row["source_document_id"] == ""
    assert row["source_stored_path"] == ""


def test_document_registry_mark_failed_stores_error(tmp_path):
    registry = DocumentRegistry(str(tmp_path / "registry.sqlite3"))

    registry.create_processing(
        document_id="doc2",
        tenant_id="default",
        knowledge_base_type="enterprise",
        original_filename="b.txt",
        stored_filename="doc2_b.txt",
        stored_path="/tmp/doc2_b.txt",
    )
    registry.mark_failed("doc2", "load failed")

    row = registry.get("doc2")
    assert row is not None
    assert row["status"] == "failed"
    assert row["error"] == "load failed"


# ---------------------------------------------------------------------------
# list_personal_ready / find_personal_ready_by_path
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_registry(tmp_path):
    reg = DocumentRegistry(str(tmp_path / "reg.sqlite3"))
    reg.init()
    return reg


def _seed(reg, *, doc_id="doc1", owner="user1", kb_type="personal",
          status="ready", filename="test.txt", stored_path="/uploads/test.txt"):
    reg.create_processing(
        document_id=doc_id, tenant_id="default", knowledge_base_type=kb_type,
        owner_user_id=owner, original_filename=filename,
        stored_filename=filename, stored_path=stored_path,
    )
    if status == "ready":
        reg.mark_ready(doc_id, chunk_count=5)
    elif status == "failed":
        reg.mark_failed(doc_id, error="处理失败")


class TestListPersonalReady:
    def test_returns_own_files(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", stored_path="/uploads/a.txt")
        _seed(seeded_registry, doc_id="d2", stored_path="/uploads/b.txt")
        files = seeded_registry.list_personal_ready("user1")
        assert len(files) == 2

    def test_excludes_other_owner(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", owner="user1", stored_path="/uploads/a.txt")
        _seed(seeded_registry, doc_id="d2", owner="user2", stored_path="/uploads/b.txt")
        files = seeded_registry.list_personal_ready("user1")
        assert len(files) == 1
        assert files[0]["stored_path"] == "/uploads/a.txt"

    def test_excludes_enterprise(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", kb_type="enterprise", stored_path="/uploads/e.txt")
        assert seeded_registry.list_personal_ready("user1") == []

    def test_excludes_processing(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", status="processing", stored_path="/uploads/p.txt")
        assert seeded_registry.list_personal_ready("user1") == []

    def test_excludes_failed(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", status="failed", stored_path="/uploads/f.txt")
        assert seeded_registry.list_personal_ready("user1") == []

    def test_ordered_by_created_at_desc(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", filename="first.txt", stored_path="/uploads/first.txt")
        _seed(seeded_registry, doc_id="d2", filename="second.txt", stored_path="/uploads/second.txt")
        files = seeded_registry.list_personal_ready("user1")
        assert files[0]["original_filename"] == "second.txt"

    def test_limit(self, seeded_registry):
        for i in range(5):
            _seed(seeded_registry, doc_id=f"d{i}", filename=f"f{i}.txt", stored_path=f"/uploads/f{i}.txt")
        assert len(seeded_registry.list_personal_ready("user1", limit=3)) == 3

    def test_empty(self, seeded_registry):
        assert seeded_registry.list_personal_ready("nobody") == []

    def test_fields(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", stored_path="/uploads/a.txt")
        f = seeded_registry.list_personal_ready("user1")[0]
        for key in ("document_id", "original_filename", "stored_filename", "stored_path", "created_at", "updated_at"):
            assert key in f


class TestFindPersonalReadyByPath:
    def test_finds_existing(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", stored_path="/uploads/a.txt")
        r = seeded_registry.find_personal_ready_by_path("user1", "/uploads/a.txt")
        assert r is not None
        assert r["owner_user_id"] == "user1"

    def test_wrong_owner(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", owner="user1", stored_path="/uploads/a.txt")
        assert seeded_registry.find_personal_ready_by_path("user2", "/uploads/a.txt") is None

    def test_enterprise(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", kb_type="enterprise", stored_path="/uploads/e.txt")
        assert seeded_registry.find_personal_ready_by_path("user1", "/uploads/e.txt") is None

    def test_nonexistent(self, seeded_registry):
        assert seeded_registry.find_personal_ready_by_path("user1", "/nope.txt") is None

    def test_processing(self, seeded_registry):
        _seed(seeded_registry, doc_id="d1", status="processing", stored_path="/uploads/p.txt")
        assert seeded_registry.find_personal_ready_by_path("user1", "/uploads/p.txt") is None


# ---------------------------------------------------------------------------
# generated copy metadata
# ---------------------------------------------------------------------------

class TestGeneratedCopyRegistry:
    def test_register_generated_copy(self, seeded_registry):
        _seed(seeded_registry, doc_id="source_doc", stored_path="/uploads/source.txt")
        source = seeded_registry.get("source_doc")

        seeded_registry.register_generated_copy(
            document_id="copy_doc",
            source_record=source,
            stored_path="/uploads/source_副本_abcd.txt",
            stored_filename="source_副本_abcd.txt",
        )

        copy = seeded_registry.get("copy_doc")
        assert copy["knowledge_base_type"] == "personal"
        assert copy["owner_user_id"] == "user1"
        assert copy["status"] == "ready"
        assert copy["is_generated_copy"] == 1
        assert copy["source_document_id"] == "source_doc"
        assert copy["source_stored_path"] == "/uploads/source.txt"

    def test_generated_copy_can_be_found_by_path(self, seeded_registry):
        _seed(seeded_registry, doc_id="source_doc", stored_path="/uploads/source.txt")
        source = seeded_registry.get("source_doc")
        seeded_registry.register_generated_copy(
            document_id="copy_doc",
            source_record=source,
            stored_path="/uploads/source_副本_abcd.txt",
            stored_filename="source_副本_abcd.txt",
        )

        assert seeded_registry.find_personal_ready_generated_copy_by_path(
            "user1",
            "/uploads/source_副本_abcd.txt",
        )["document_id"] == "copy_doc"
        assert seeded_registry.find_personal_ready_generated_copy_by_path("user2", "/uploads/source_副本_abcd.txt") is None
        assert seeded_registry.find_personal_ready_generated_copy_by_path("user1", "/uploads/source.txt") is None

    def test_migrates_old_schema_with_default_copy_fields(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "old.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE documents (
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
            """
            INSERT INTO documents (
                document_id, tenant_id, knowledge_base_type, owner_user_id,
                original_filename, stored_filename, stored_path, status, created_at, updated_at
            ) VALUES ('old_doc', 'default', 'personal', 'user1', 'old.txt', 'old.txt', '/uploads/old.txt', 'ready', 't1', 't1')
            """
        )
        conn.commit()
        conn.close()

        reg = DocumentRegistry(str(db_path))
        row = reg.get("old_doc")

        assert row["is_generated_copy"] == 0
        assert row["source_document_id"] == ""
        assert row["source_stored_path"] == ""
