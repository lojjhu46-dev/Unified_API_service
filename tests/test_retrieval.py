"""检索模块测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.retrieval.retriever import Retriever
from app.retrieval.ingest import validate_file_extension, ingest_file
from app.schemas import SourceItem


@pytest.fixture
def mock_retriever():
    retriever = Retriever()
    retriever._use_chroma = False
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


@patch("app.retrieval.vector_store.add_documents")
@patch("app.retrieval.ingest.split_documents")
@patch("app.retrieval.ingest.load_document")
def test_ingest_file(mock_load, mock_split, mock_add):
    mock_load.return_value = [MagicMock()]
    mock_split.return_value = [MagicMock(), MagicMock()]
    mock_add.return_value = 2

    result = ingest_file("/tmp/test.pdf")
    assert result["chunks"] == 2
    assert "document_id" in result
    assert "filename" in result


@pytest.mark.asyncio
async def test_chroma_search_fallback_to_mock():
    retriever = Retriever()
    retriever._use_chroma = False
    results = await retriever.search("测试", top_k=2)
    assert len(results) > 0
