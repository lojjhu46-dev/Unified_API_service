from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.tools.summarize import summarize_uploaded_file_content


@pytest.mark.asyncio
async def test_summarize_uploaded_file_rejects_unsupported_extension():
    result = await summarize_uploaded_file_content(b"hello", "image.png")

    assert result["success"] is False
    assert "PDF" in result["error"]


@pytest.mark.asyncio
async def test_summarize_uploaded_file_uses_extracted_text():
    documents = [
        SimpleNamespace(page_content="第一段内容", metadata={"page": 1}),
        SimpleNamespace(page_content="第二段内容", metadata={"page": 2}),
    ]

    with patch("app.tools.summarize.load_document", return_value=documents) as mock_load, \
         patch("app.tools.summarize._generate_summary", new=AsyncMock(return_value="总结结果")) as mock_generate:
        result = await summarize_uploaded_file_content(b"hello", "report.txt")

    assert result["success"] is True
    assert result["resource_type"] == "txt"
    assert result["title"] == "report.txt"
    assert result["summary"] == "总结结果"
    assert result["outline"] == ["1", "2"]
    assert "第一段内容" in result["sample_text"]
    mock_load.assert_called_once()
    mock_generate.assert_awaited_once()
