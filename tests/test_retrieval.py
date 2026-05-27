"""检索模块测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.retrieval.retriever import Retriever
from app.config import settings
from app.retrieval.ingest import (
    ingest_file,
    sanitize_filename,
    save_uploaded_file,
    validate_file_extension,
)
from app.schemas import SourceItem


@pytest.fixture
def mock_retriever():
    retriever = Retriever()
    retriever._use_chroma = False
    retriever.refresh = MagicMock(return_value=False)
    return retriever


@pytest.mark.asyncio
async def test_mock_search(mock_retriever):
    results = await mock_retriever.search("测试问题", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, SourceItem) for r in results)
    assert results[0].score > results[1].score


def test_validate_file_extension():
    assert validate_file_extension("test.pdf") is True
    assert validate_file_extension("test.txt") is True
    assert validate_file_extension("test.doc") is False
    assert validate_file_extension("test.xlsx") is False


def test_sanitize_filename_removes_paths_and_invalid_chars():
    assert sanitize_filename("../evil.txt") == "evil.txt"
    assert sanitize_filename(r"C:\tmp\evil.txt") == "evil.txt"
    assert sanitize_filename('bad:name?.pdf') == "bad_name_.pdf"


def test_save_uploaded_file_stays_inside_upload_dir(tmp_path):
    old_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path)
    try:
        for filename in ["../evil.txt", r"C:\tmp\evil.txt", "bad:name?.txt"]:
            saved_path = save_uploaded_file(b"hello", filename)
            resolved = __import__("pathlib").Path(saved_path).resolve()
            assert tmp_path.resolve() in resolved.parents
            assert ".." not in resolved.name
            assert ":" not in resolved.name
    finally:
        settings.upload_dir = old_upload_dir


@patch("app.retrieval.vector_store.add_documents")
@patch("app.retrieval.ingest.split_documents")
@patch("app.retrieval.ingest.load_document")
def test_ingest_file(mock_load, mock_split, mock_add):
    mock_load.return_value = [MagicMock()]
    chunk1 = MagicMock(metadata={"page": 1})
    chunk2 = MagicMock(metadata={"page": 2})
    mock_split.return_value = [chunk1, chunk2]
    mock_add.return_value = 2

    result = ingest_file("/tmp/stored.pdf", original_filename="../test.pdf")
    assert result["chunks"] == 2
    assert "document_id" in result
    assert "filename" in result
    assert result["filename"] == "test.pdf"
    assert chunk1.metadata["document_id"] == result["document_id"]
    assert chunk2.metadata["document_id"] == result["document_id"]
    assert chunk1.metadata["chunk_index"] == 0
    assert chunk2.metadata["chunk_index"] == 1
    assert chunk1.metadata["original_filename"] == "test.pdf"
    assert chunk1.metadata["stored_filename"] == "stored.pdf"


@pytest.mark.asyncio
async def test_chroma_search_fallback_to_mock():
    retriever = Retriever()
    retriever._use_chroma = False
    retriever.refresh = MagicMock(return_value=False)
    results = await retriever.search("测试", top_k=2)
    assert len(results) > 0


def test_normalize_distance_score():
    retriever = Retriever()
    assert retriever._normalize_distance_score(0) == 1.0
    assert 0.0 < retriever._normalize_distance_score(1.5) < 1.0
    assert retriever._normalize_distance_score(None) == 0.0
    assert 0.0 <= retriever._normalize_distance_score(-1) <= 1.0


@pytest.mark.asyncio
async def test_chroma_exception_falls_back_to_mock():
    retriever = Retriever()
    retriever._use_chroma = True
    with patch.object(retriever, "_chroma_search", new=AsyncMock(side_effect=RuntimeError("boom"))):
        results = await retriever.search("测试", top_k=2)

    assert len(results) == 2
    assert all(item.source_type == "knowledge_base" for item in results)
