"""Document registry tests."""

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
