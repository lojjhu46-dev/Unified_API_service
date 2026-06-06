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
