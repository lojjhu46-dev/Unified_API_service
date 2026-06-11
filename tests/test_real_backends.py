"""真实文档后端测试

覆盖路径安全、TXT/PDF 真实后端、DOCX/XLSX HTTP MCP mock。
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.documents.adapters.pdf_real import RealPdfBackend
from app.documents.adapters.txt_real import RealTxtBackend
from app.documents.adapters.docx_http import HttpDocxBackend
from app.documents.adapters.xlsx_http import HttpXlsxBackend
from app.documents.models import BackendType, FileType
from app.documents.path_security import validate_file_path, get_allowed_dirs
from app.documents.tools import (
    _reset_executor,
    document_extract,
    document_plan,
    document_apply_plan,
    document_review,
    set_executor,
)
from app.documents.executor import DocumentOperationAgent
from app.config import settings


# ---------------------------------------------------------------------------
# 路径安全
# ---------------------------------------------------------------------------

class TestPathSecurity:
    def test_allowed_dir_file_passes(self, tmp_path):
        """上传目录内文件允许访问"""
        with patch.object(settings, "upload_dir", str(tmp_path)):
            f = tmp_path / "test.txt"
            f.write_text("hello")
            resolved, err = validate_file_path(str(f))
            assert err is None
            assert resolved == f.resolve()

    def test_outside_dir_rejected(self, tmp_path):
        """目录外文件被拒绝"""
        with patch.object(settings, "upload_dir", str(tmp_path / "uploads")):
            (tmp_path / "uploads").mkdir()
            f = tmp_path / "outside.txt"
            f.write_text("hello")
            _, err = validate_file_path(str(f))
            assert err is not None
            assert "不在允许的目录内" in err

    def test_nonexistent_file_rejected(self, tmp_path):
        """不存在的文件被拒绝"""
        with patch.object(settings, "upload_dir", str(tmp_path)):
            _, err = validate_file_path(str(tmp_path / "nope.txt"))
            assert err is not None
            assert "不存在" in err

    def test_path_traversal_rejected(self, tmp_path):
        """路径穿越被拒绝"""
        with patch.object(settings, "upload_dir", str(tmp_path / "uploads")):
            (tmp_path / "uploads").mkdir()
            _, err = validate_file_path(str(tmp_path / "uploads" / ".." / "etc" / "passwd"))
            assert err is not None

    def test_symlink_escape_rejected(self, tmp_path):
        """符号链接逃逸被拒绝"""
        with patch.object(settings, "upload_dir", str(tmp_path / "uploads")):
            uploads = tmp_path / "uploads"
            uploads.mkdir()
            target = tmp_path / "secret.txt"
            target.write_text("secret")
            link = uploads / "link.txt"
            link.symlink_to(target)
            _, err = validate_file_path(str(link))
            # 符号链接 resolve 后指向 tmp_path/secret.txt，不在 uploads 内
            assert err is not None


# ---------------------------------------------------------------------------
# 真实 TXT adapter
# ---------------------------------------------------------------------------

class TestRealTxtBackend:
    @pytest.fixture
    def backend(self):
        return RealTxtBackend()

    @pytest.mark.asyncio
    async def test_read_structure(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行\n第三行")
        structure = await backend.read_structure(str(f))
        assert structure["type"] == "txt"
        assert structure["line_count"] == 3

    @pytest.mark.asyncio
    async def test_read_lines(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行\n第三行")
        lines = await backend.read_lines(str(f))
        assert lines == ["第一行", "第二行", "第三行"]

    @pytest.mark.asyncio
    async def test_replace_line(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行\n第三行")
        await backend.replace_line(str(f), 2, "新第二行")
        lines = await backend.read_lines(str(f))
        assert lines[1] == "新第二行"

    @pytest.mark.asyncio
    async def test_delete_line(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行\n第三行")
        await backend.delete_line(str(f), 2)
        lines = await backend.read_lines(str(f))
        assert len(lines) == 2
        assert lines[1] == "第三行"

    @pytest.mark.asyncio
    async def test_append_lines(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("第一行")
        await backend.append_lines(str(f), ["第二行", "第三行"])
        lines = await backend.read_lines(str(f))
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_replace_text(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("实验体会的内容\n其他内容")
        count = await backend.replace_text(str(f), "实验体会", "研究心得")
        assert count == 1
        lines = await backend.read_lines(str(f))
        assert "研究心得" in lines[0]

    @pytest.mark.asyncio
    async def test_replace_text_no_match(self, backend, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("内容")
        count = await backend.replace_text(str(f), "不存在", "新值")
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_copy(self, backend, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("内容")
        dst = tmp_path / "dst.txt"
        await backend.save_copy(str(src), str(dst))
        assert dst.read_text() == "内容"
        assert src.read_text() == "内容"  # 原文件不变

    @pytest.mark.asyncio
    async def test_file_not_found(self, backend, tmp_path):
        with pytest.raises(FileNotFoundError):
            await backend.read_lines(str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# 真实 PDF reader
# ---------------------------------------------------------------------------

class TestRealPdfBackend:
    @pytest.fixture
    def backend(self):
        return RealPdfBackend()

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """创建一个简单的 PDF 文件"""
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        f = tmp_path / "test.pdf"
        with open(f, "wb") as fp:
            writer.write(fp)
        return f

    @pytest.mark.asyncio
    async def test_read_structure(self, backend, sample_pdf):
        structure = await backend.read_structure(str(sample_pdf))
        assert structure["type"] == "pdf"
        assert structure["page_count"] == 2

    @pytest.mark.asyncio
    async def test_read_page(self, backend, sample_pdf):
        text = await backend.read_page(str(sample_pdf), 1)
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_extract_text_range(self, backend, sample_pdf):
        text = await backend.extract_text(str(sample_pdf), 1, 2)
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_page_out_of_range(self, backend, sample_pdf):
        with pytest.raises(IndexError):
            await backend.read_page(str(sample_pdf), 99)

    @pytest.mark.asyncio
    async def test_file_not_found(self, backend, tmp_path):
        with pytest.raises(FileNotFoundError):
            await backend.read_structure(str(tmp_path / "nope.pdf"))


# ---------------------------------------------------------------------------
# DOCX HTTP MCP mock
# ---------------------------------------------------------------------------

class TestHttpDocxBackend:
    @pytest.fixture
    def backend(self):
        return HttpDocxBackend("http://localhost:9000", timeout=5.0)

    @pytest.mark.asyncio
    async def test_read_structure_success(self, backend):
        mock_resp = httpx.Response(200, json={
            "success": True,
            "data": {"type": "docx", "paragraph_count": 3},
        })
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            structure = await backend.read_structure("/tmp/test.docx")
        assert structure["paragraph_count"] == 3

    @pytest.mark.asyncio
    async def test_server_error(self, backend):
        mock_resp = httpx.Response(500, text="Internal Server Error")
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            from app.documents.adapters.base import BackendUnavailableError
            with pytest.raises(BackendUnavailableError):
                await backend.read_structure("/tmp/test.docx")

    @pytest.mark.asyncio
    async def test_timeout(self, backend):
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
            from app.documents.adapters.base import BackendUnavailableError
            with pytest.raises(BackendUnavailableError):
                await backend.read_structure("/tmp/test.docx")

    @pytest.mark.asyncio
    async def test_timeout_then_recovery(self, backend):
        """临时超时应在本次调用内短重试并恢复"""
        ok_resp = httpx.Response(200, json={"success": True, "data": {"type": "docx", "paragraph_count": 5}})
        post = AsyncMock(side_effect=[
            httpx.TimeoutException("timeout"),
            ok_resp,
        ])
        with patch("httpx.AsyncClient.post", new=post), patch("asyncio.sleep", new=AsyncMock()) as sleep:
            structure = await backend.read_structure("/tmp/test.docx")
        assert structure["paragraph_count"] == 5
        assert post.await_count == 2
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_false(self, backend):
        mock_resp = httpx.Response(200, json={
            "success": False,
            "error": "文件不存在",
        })
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            from app.documents.adapters.base import BackendUnavailableError
            with pytest.raises(BackendUnavailableError):
                await backend.read_structure("/tmp/test.docx")

    @pytest.mark.asyncio
    async def test_non_json_response(self, backend):
        mock_resp = httpx.Response(200, text="not json")
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            from app.documents.adapters.base import BackendUnavailableError
            with pytest.raises(BackendUnavailableError):
                await backend.read_structure("/tmp/test.docx")


# ---------------------------------------------------------------------------
# XLSX HTTP MCP mock
# ---------------------------------------------------------------------------

class TestHttpXlsxBackend:
    @pytest.fixture
    def backend(self):
        return HttpXlsxBackend("http://localhost:9001", timeout=5.0)

    @pytest.mark.asyncio
    async def test_read_cell_success(self, backend):
        mock_resp = httpx.Response(200, json={
            "success": True,
            "data": {"value": "张三"},
        })
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            value = await backend.read_cell("/tmp/test.xlsx", "Sheet1", "A1")
        assert value == "张三"

    @pytest.mark.asyncio
    async def test_modify_cell_success(self, backend):
        mock_resp = httpx.Response(200, json={"success": True, "data": {}})
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            await backend.modify_cell("/tmp/test.xlsx", "Sheet1", "A1", "新值")

    @pytest.mark.asyncio
    async def test_timeout_then_recovery(self, backend):
        ok_resp = httpx.Response(200, json={"success": True, "data": {"value": "张三"}})
        post = AsyncMock(side_effect=[
            httpx.ConnectTimeout("timeout"),
            ok_resp,
        ])
        with patch("httpx.AsyncClient.post", new=post), patch("asyncio.sleep", new=AsyncMock()) as sleep:
            value = await backend.read_cell("/tmp/test.xlsx", "Sheet1", "A1")
        assert value == "张三"
        assert post.await_count == 2
        sleep.assert_awaited_once()
